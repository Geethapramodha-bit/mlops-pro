import csv
import os
from datetime import datetime

CONVERSATION_PATH = "data/conversations/conversations.csv"


def log_conversation(
    user_query: str,
    intent: str,
    response: str,
    feedback: int = 0
):
    file_exists = os.path.exists(CONVERSATION_PATH)

    with open(
    CONVERSATION_PATH,
    "a",
    newline="",
    encoding="utf-8"
) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "user_query",
                "intent",
                "response",
                "feedback"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_query,
            intent,
            response,
            feedback
        ])