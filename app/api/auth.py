
import re

from fastapi import APIRouter, Depends, HTTPException, Response, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.services.database import Base, engine, get_db
from app.services.models import User
from app.services.auth_service import (
    authenticate_user,
    create_session_token,
    get_session_user_id,
    hash_password,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Create the tables defined in models.py.
Base.metadata.create_all(bind=engine)

COOKIE_NAME = "botplayer_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    token = request.cookies.get(COOKIE_NAME)

    if not token:
        raise HTTPException(status_code=401, detail="Please log in.")

    user_id = get_session_user_id(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Account not found.")

    return user


@router.post("/signup", status_code=201)
def signup(
    payload: SignupRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    name = payload.name.strip()
    email = str(payload.email).strip().lower()

    if len(name) < 2:
        raise HTTPException(status_code=422, detail="Name must contain at least two characters.")

    if len(payload.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=422, detail="Password is too long.")

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(payload.password),
    )

    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    response.set_cookie(
        key=COOKIE_NAME,
        value=create_session_token(user.id),
        httponly=True,
        secure=False,  # Set True when deployed behind HTTPS.
        samesite="lax",
        max_age=COOKIE_MAX_AGE,
        path="/",
    )

    return {
        "message": "Account created successfully.",
        "user": {"id": user.id, "name": user.name, "email": user.email},
    }


@router.post("/login")
def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    user = authenticate_user(db, str(payload.email), payload.password)

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    response.set_cookie(
        key=COOKIE_NAME,
        value=create_session_token(user.id),
        httponly=True,
        secure=False,  # Set True when deployed behind HTTPS.
        samesite="lax",
        max_age=COOKIE_MAX_AGE,
        path="/",
    )

    return {
        "message": "Logged in successfully.",
        "user": {"id": user.id, "name": user.name, "email": user.email},
    }


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"user": {"id": user.id, "name": user.name, "email": user.email}}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(
        key=COOKIE_NAME,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
    )
    return {"message": "Logged out successfully."}
