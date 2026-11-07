import getpass
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend.auth import USERNAME_RE, hash_password  # noqa: E402


def main() -> None:
    username = input("Username: ").strip()
    if not USERNAME_RE.match(username):
        sys.exit("Username must be 3-64 characters: letters, digits, _ . @ -")
    password = getpass.getpass("Password (leave empty to generate one): ")
    if not password:
        password = secrets.token_urlsafe(16)
        print(f"Generated password: {password}")
    if len(password) < 10:
        sys.exit("Use at least 10 characters.")
    print("\nAdd this to AUTH_USERS (comma-separate several accounts):")
    print(f"{username}:{hash_password(password)}")
    print("\nOptional API key for /api/research (set as API_KEY):")
    print(secrets.token_urlsafe(32))


if __name__ == "__main__":
    main()
