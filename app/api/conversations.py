
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.services.database import get_db
from app.services.models import Conversation, Message, User

router = APIRouter(prefix="/conversations", tags=["Conversations"])


class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=255)
    assistant_mode: str = Field(default="agentic", pattern="^(agentic|support)$")


class MessageCreate(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=50000)


def get_owned_conversation(
    conversation_id: int,
    user_id: int,
    db: Session,
) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        .first()
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return conversation


def serialize_message(message: Message):
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
    }


def serialize_conversation(conversation: Conversation):
    return {
        "id": conversation.id,
        "title": conversation.title,
        "assistant_mode": conversation.assistant_mode,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


@router.get("")
def list_conversations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )

    return {
        "conversations": [
            serialize_conversation(conversation)
            for conversation in conversations
        ]
    }


@router.post("", status_code=201)
def create_conversation(
    payload: ConversationCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = Conversation(
        user_id=user.id,
        title=payload.title.strip() or "New chat",
        assistant_mode=payload.assistant_mode,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return {"conversation": serialize_conversation(conversation)}


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = get_owned_conversation(
        conversation_id,
        user.id,
        db,
    )

    return {
        "conversation": serialize_conversation(conversation),
        "messages": [
            serialize_message(message)
            for message in conversation.messages
        ],
    }


@router.post("/{conversation_id}/messages", status_code=201)
def add_message(
    conversation_id: int,
    payload: MessageCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = get_owned_conversation(
        conversation_id,
        user.id,
        db,
    )

    message = Message(
        conversation_id=conversation.id,
        role=payload.role,
        content=payload.content,
    )

    db.add(message)

    if payload.role == "user" and conversation.title == "New chat":
        conversation.title = payload.content.strip()[:60] or "New chat"

    db.commit()
    db.refresh(message)
    db.refresh(conversation)

    return {
        "message": serialize_message(message),
        "conversation": serialize_conversation(conversation),
    }


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = get_owned_conversation(
        conversation_id,
        user.id,
        db,
    )

    db.delete(conversation)
    db.commit()

    return None
