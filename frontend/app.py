"""Chainlit UI. Runs the LangGraph pipeline in-process and streams agent activity live."""

from __future__ import annotations

import logging
import re
from datetime import datetime

import chainlit as cl

from ai_core.config import get_settings
from ai_core.graph import run_research
from ai_core.models import ResearchReport
from frontend import auth

log = logging.getLogger("frontend")
_data_layer = None

AGENT_ICONS = {
    "Orchestrator": "🧭",
    "Planner": "🗂️",
    "Writer": "✍️",
    "Fact-checker": "🔎",
    "Editor": "📎",
    "System": "⚠️",
}
KIND_MARK = {"result": "✅", "warn": "⚠️", "error": "❌", "info": "•"}


# ------------------------------------------------------------------ persistence & auth
@cl.data_layer
def get_data_layer():
    global _data_layer
    if _data_layer is None:
        import json

        from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

        class SQLiteDataLayer(SQLAlchemyDataLayer):
            """Chainlit's layer targets Postgres arrays; SQLite can't bind Python lists (e.g. the
            chat-profile tag on new threads), which silently broke thread history in v1."""

            async def execute_sql(self, query, parameters):
                parameters = {
                    k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in (parameters or {}).items()
                }
                result = await super().execute_sql(query, parameters)
                if isinstance(result, list):
                    for row in result:
                        for key in ("tags", "thread_tags", "step_tags"):
                            if isinstance(row.get(key), str) and row[key].startswith("["):
                                try:
                                    row[key] = json.loads(row[key])
                                except ValueError:
                                    pass
                return result

        _data_layer = SQLiteDataLayer(conninfo=get_settings().database_url)
    return _data_layer


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    username = username.strip()
    if not auth.USERNAME_RE.match(username) or len(password) < 6:
        return None
    dl = get_data_layer()
    stored = await auth.get_hash(username)
    if stored:
        if not auth.verify_password(password, stored):
            return None
        return await dl.get_user(username) or await dl.create_user(cl.User(identifier=username, metadata={"role": "user"}))

    existing = await dl.get_user(username)
    if existing:
        legacy = (existing.metadata or {}).get("password")
        if legacy is not None:
            # v1 account: plaintext password lived in metadata. Accept only an exact match, then migrate.
            if legacy != password:
                return None
            await auth.set_hash(username, password)
            return existing
        # A user row with no credentials at all: Chainlit re-creates the row from a still-valid
        # session cookie after the database was reset (e.g. a redeploy on an ephemeral disk).
        # Let the person claim it again instead of locking them out forever.
        if not get_settings().allow_signup:
            return None
        await auth.set_hash(username, password)
        return existing

    if not get_settings().allow_signup:
        return None
    await auth.set_hash(username, password)
    return await dl.create_user(cl.User(identifier=username, metadata={"role": "user"}))


# ------------------------------------------------------------------ chat setup
@cl.set_chat_profiles
async def chat_profiles(current_user: cl.User | None = None, language: str | None = None):
    return [
        cl.ChatProfile(
            name="Quick research",
            markdown_description="3 expert agents · 1 search round · ~1 minute",
            icon="https://api.iconify.design/lucide/zap.svg?color=%238b86f5",
            default=True,
        ),
        cl.ChatProfile(
            name="Deep research",
            markdown_description="4 expert agents · 2 search rounds with gap-filling · fact-check revision · ~3 minutes",
            icon="https://api.iconify.design/lucide/telescope.svg?color=%238b86f5",
        ),
    ]


@cl.set_starters
async def starters(current_user: cl.User | None = None, language: str | None = None):
    return [
        cl.Starter(
            label="Solid-state batteries",
            message="What is the current state of solid-state battery technology and what is blocking mass production?",
            icon="https://api.iconify.design/lucide/battery-charging.svg?color=%238b86f5",
        ),
        cl.Starter(
            label="GLP-1 drugs",
            message="What does the evidence say about the long-term effects of GLP-1 receptor agonists like semaglutide?",
            icon="https://api.iconify.design/lucide/pill.svg?color=%238b86f5",
        ),
        cl.Starter(
            label="Small language models",
            message="How do small language models compare to large ones for on-device AI, and what techniques close the gap?",
            icon="https://api.iconify.design/lucide/cpu.svg?color=%238b86f5",
        ),
        cl.Starter(
            label="Microplastics",
            message="What is known about microplastics in the human body and their health effects?",
            icon="https://api.iconify.design/lucide/microscope.svg?color=%238b86f5",
        ),
    ]


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    pass


# ------------------------------------------------------------------ helpers
async def read_upload(element) -> tuple[str, str]:
    name = getattr(element, "name", "document") or "document"
    path = getattr(element, "path", None)
    if not path:
        return "", name
    if name.lower().endswith(".pdf") or (getattr(element, "mime", "") or "").endswith("pdf"):
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages), name
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read(), name


def render_report(report: ResearchReport) -> str:
    def link(m: re.Match) -> str:
        n = int(m.group(1))
        ref = next((r for r in report.references if r.id == n), None)
        if ref and ref.url.startswith("http"):
            title = ref.title.replace('"', "'")
            return f'[[{n}]]({ref.url} "{title}")'
        return f"[{n}]"

    body = re.sub(r"\[(\d+)\]", link, report.markdown)
    md = f"# {report.title}\n\n{body}"
    if report.references:
        md += "\n\n---\n\n### References\n\n"
        for r in report.references:
            label = f"[{r.title}]({r.url})" if r.url.startswith("http") else r.title
            md += f"{r.id}. {label} — *{r.origin}*\n"
    return md


def plain_markdown(report: ResearchReport) -> str:
    md = f"# {report.title}\n\n{report.markdown}\n"
    if report.references:
        md += "\n## References\n\n" + "".join(f"{r.id}. {r.title}. {r.url}\n" for r in report.references)
    return md


# ------------------------------------------------------------------ main handler
@cl.on_message
async def on_message(message: cl.Message):
    topic = message.content.strip()
    document_text, document_name = "", "document"
    for el in message.elements or []:
        try:
            text, document_name = await read_upload(el)
            document_text += text + "\n"
        except Exception as e:
            await cl.Message(content=f"⚠️ Couldn't read **{getattr(el, 'name', 'file')}**: {e}").send()
    if document_text.strip() and not topic:
        topic = f"Summarise and analyse the key findings of {document_name}"
    if len(topic) < 3:
        await cl.Message(content="Please describe a research topic (a few words or a question).").send()
        return

    profile = cl.user_session.get("chat_profile") or "Quick research"
    depth = "deep" if profile.startswith("Deep") else "quick"
    started = datetime.now()

    tasks = cl.TaskList(status="Researching…")
    stages = {
        "Orchestrator": cl.Task(title="Assemble expert panel", status=cl.TaskStatus.RUNNING),
        "research": cl.Task(title="Search & extract evidence"),
        "Planner": cl.Task(title="Plan the outline"),
        "Writer": cl.Task(title="Write the report"),
        "Fact-checker": cl.Task(title="Fact-check against evidence"),
        "Editor": cl.Task(title="Verify citations"),
    }
    for t in stages.values():
        await tasks.add_task(t)
    await tasks.send()

    order = list(stages)

    async def advance(key: str):
        idx = order.index(key)
        for i, k in enumerate(order):
            if i < idx and stages[k].status != cl.TaskStatus.DONE:
                stages[k].status = cl.TaskStatus.DONE
        if stages[key].status != cl.TaskStatus.DONE:
            stages[key].status = cl.TaskStatus.RUNNING
        await tasks.send()

    report: ResearchReport | None = None
    async with cl.Step(name=f"{profile} · agents at work", type="run", default_open=True) as step:
        step.output = ""
        async for kind, payload in run_research(topic, depth, document_text, document_name):
            if kind == "report":
                report = payload
                continue
            agent = payload.get("agent", "Agent")
            stage = agent if agent in stages else "research"
            if agent != "System":
                await advance(stage)
            icon = AGENT_ICONS.get(agent, "🧑‍🔬")
            mark = KIND_MARK.get(payload.get("kind", "info"), "•")
            await step.stream_token(f"{mark} {icon} **{agent}** — {payload.get('message', '')}\n\n")
        elapsed = (datetime.now() - started).seconds
        step.name = f"{profile} · finished in {elapsed // 60}m {elapsed % 60}s"

    ok = report is not None and bool(report.references)
    for t in stages.values():
        t.status = cl.TaskStatus.DONE if ok else (t.status if t.status == cl.TaskStatus.DONE else cl.TaskStatus.FAILED)
    tasks.status = "Done" if ok else "Failed"
    await tasks.send()

    if report is None:
        await cl.Message(content="No report was produced.").send()
        return

    slug = re.sub(r"[^a-z0-9]+", "-", report.title.lower()).strip("-")[:60] or "report"
    elements = []
    actions = []
    if ok:
        elements.append(
            cl.File(name=f"{slug}.md", content=plain_markdown(report).encode(), display="inline", mime="text/markdown")
        )
        if depth == "quick":
            actions.append(
                cl.Action(
                    name="go_deeper",
                    payload={"topic": topic},
                    label="Go deeper",
                    tooltip="Re-run with 4 agents, 2 search rounds and a fact-check revision",
                    icon="telescope",
                )
            )
    await cl.Message(content=render_report(report), elements=elements, actions=actions).send()


@cl.action_callback("go_deeper")
async def go_deeper(action: cl.Action):
    cl.user_session.set("chat_profile", "Deep research")
    await action.remove()
    msg = cl.Message(content=action.payload["topic"], author="User", type="user_message")
    await msg.send()
    await on_message(msg)
