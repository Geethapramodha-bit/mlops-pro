import csv
import os
from datetime import datetime

CONVERSATION_PATH = "data/conversations/conversations.csv"


def get_next_conversation_id():
    if not os.path.exists(CONVERSATION_PATH):
        return 1

    with open(
        CONVERSATION_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return 1

    ids = []

    for row in rows:
        if row.get("conversation_id"):
            ids.append(int(row["conversation_id"]))

    return max(ids, default=0) + 1


def log_conversation(
    user_query: str,
    intent: str,
    response: str
):
    conversation_id = get_next_conversation_id()

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
                "conversation_id",
                "timestamp",
                "user_query",
                "intent",
                "response",
                "feedback"
            ])

        writer.writerow([
            conversation_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_query,
            intent,
            response,
            ""
        ])

    return conversation_id


def update_feedback(
    conversation_id: int,
    feedback: int
):
    rows = []

    with open(
        CONVERSATION_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        for row in reader:

            if int(row["conversation_id"]) == conversation_id:
                row["feedback"] = str(feedback)

            rows.append(row)

    with open(
        CONVERSATION_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    return True