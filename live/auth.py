"""The secret token that lets the live call page use the voice server.

Only used when the app is online (APP_PASSWORD is set). The Streamlit app gives the token to the live page
after the student typed the password; the voice server refuses /session and /tool without it, so nobody can
get an OpenAI key from the server just by knowing its address.
"""
import hashlib
import hmac
import os


def live_token() -> str | None:
    """The same token in the app and the voice server, made from APP_PASSWORD. None on the laptop."""
    password = os.getenv("APP_PASSWORD")
    if not password:
        return None
    return hmac.new(password.encode(), b"study-coach-live-call", hashlib.sha256).hexdigest()


def token_ok(token: str | None) -> bool:
    """True when no password is set (laptop), or when the token is the right one."""
    expected = live_token()
    if expected is None:
        return True
    return token is not None and hmac.compare_digest(token.encode(), expected.encode())
