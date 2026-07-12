import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chainlit as cl
import json
import urllib.parse
import aiohttp
import os

data_layer = SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///chainlit.db")


@cl.data_layer
def get_data_layer():
    return data_layer


import uuid


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    print(f"[AUTH] Authentication attempt for username: '{username}'")
    user = await data_layer.get_user(identifier=username)
    if user:
        stored_password = user.metadata.get("password")
        if stored_password is None:
            print(f"[AUTH] Legacy user '{username}' logged in. Password accepted.")
            return user
        elif stored_password == password:
            print(f"[AUTH] Successful login for '{username}'")
            return user
        else:
            print(f"[AUTH] Password mismatch for '{username}'. Rejecting login.")
            return None
    else:
        print(f"[AUTH] Auto-registering new user '{username}'")
        user = await data_layer.create_user(
            cl.User(
                identifier=username, metadata={"role": "user", "password": password}
            )
        )
        return user


@cl.on_chat_start
async def on_chat_start():
    pass


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    print(f"[CHAT] Resumed previous chat session: {thread['id']}")
    pass


@cl.on_message
async def on_message(message: cl.Message):
    print(f"[CHAT] Received message: '{message.content[:50]}...'")
    topic = message.content
    document_text = ""
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                if element.path.endswith(".pdf"):
                    try:
                        import pypdf

                        reader = pypdf.PdfReader(element.path)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                document_text += text + "\n"
                    except Exception as e:
                        await cl.Message(content=f"Error reading PDF: {e}").send()
                else:
                    try:
                        with open(element.path, "r", encoding="utf-8") as f:
                            document_text += f.read()
                    except Exception as e:
                        await cl.Message(content=f"Error reading text file: {e}").send()
    port = os.environ.get("PORT", 8000)
    api_url = f"http://localhost:{port}/api/research/stream"
    async with cl.Step(name="Agents Orchestrating") as step:
        step.output = ""
        final_report = ""
        final_references = []
        async with aiohttp.ClientSession() as session:
            try:
                payload = {"topic": topic, "document_text": document_text}
                async with session.post(api_url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if line:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line.split("data: ", 1)[1]
                                if data_str == "Finished":
                                    step.status = "done"
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    if "message" in data_json:
                                        step.output += f"✓ {data_json['message']}\n"
                                        await step.update()
                                    if "markdown" in data_json:
                                        final_report = data_json["markdown"]
                                        if "references" in data_json:
                                            final_references = data_json["references"]
                                except json.JSONDecodeError:
                                    pass
            except Exception as e:
                step.status = "error"
                step.output = f"Error connecting to FastAPI: {e}"
                await step.update()
                await cl.Message(
                    content="Failed to connect to the backend server. Is Uvicorn running?",
                    author="Assistant",
                ).send()
                return
    if final_report:
        for ref in final_references:
            ref_id = str(ref.get("id", ""))
            title = (
                ref.get("title", "Source").replace("'", "&apos;").replace('"', "&quot;")
            )
            url = ref.get("url", "#")
            if not url.startswith("http"):
                url = "#"
            md_link = f'[[{ref_id}]]({url} "{title}")'
            final_report = final_report.replace(f"[{ref_id}]", md_link)
        if final_references:
            final_report += "\n\n---\n### References\n"
            for ref in final_references:
                ref_id = str(ref.get("id", ""))
                title = (
                    ref.get("title", "Source")
                    .replace("'", "&apos;")
                    .replace('"', "&quot;")
                )
                url = ref.get("url", "#")
                if not url.startswith("http"):
                    url = "#"
                final_report += f"{ref_id}. [{title}]({url})\n"
        await cl.Message(content=final_report, author="Assistant").send()
    else:
        await cl.Message(
            content="No report was generated by the backend.", author="Assistant"
        ).send()
