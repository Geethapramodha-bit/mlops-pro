
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from itsdangerous import BadSignature, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from app.services.models import User

SESSION_SECRET = os.getenv("BOTPLAYER_SESSION_SECRET")

if not SESSION_SECRET:
    raise RuntimeError(
        "BOTPLAYER_SESSION_SECRET is missing. "
        "Set it in your environment before starting FastAPI."
    )

serializer = URLSafeTimedSerializer(
    SESSION_SECRET,
    salt="botplayer-auth-session",
)

SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def hash_password(password: str) -> str:
    # bcrypt supports at most 72 bytes; reject longer passwords.
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        raise ValueError("Password must be no longer than 72 UTF-8 bytes.")

    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        return False

    try:
        return bcrypt.checkpw(
            password_bytes,
            password_hash.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


def create_session_token(user_id: int) -> str:
    return serializer.dumps({"user_id": user_id})


def get_session_user_id(token: str) -> int | None:
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        user_id = data.get("user_id")
        return user_id if isinstance(user_id, int) and user_id > 0 else None
    except (BadSignature, TypeError, ValueError):
        return None


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    normalized_email = email.strip().lower()
    user = db.query(User).filter(User.email == normalized_email).first()

    if user is None or not verify_password(password, user.password_hash):
        return None

    return user
