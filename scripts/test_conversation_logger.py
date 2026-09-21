from app.services.conversation_logger import (
    log_conversation,
    update_feedback
)


conversation_id = log_conversation(
    user_query="Can I get a refund?",
    intent="refund",
    response="Please provide your order or transaction details."
)

print("Created conversation:", conversation_id)

update_feedback(
    conversation_id=conversation_id,
    feedback=1
)

print("Feedback updated successfully!")