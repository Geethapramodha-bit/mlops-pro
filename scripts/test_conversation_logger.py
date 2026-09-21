from app.services.conversation_logger import log_conversation


log_conversation(
    user_query="I forgot my password",
    intent="account",
    response="You can reset your password using the Forgot Password option.",
    feedback=1
)

print("Conversation logged successfully!")
