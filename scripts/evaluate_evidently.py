import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


CONVERSATION_PATH = "data/conversations/conversations.csv"
REPORT_PATH = "reports/evidently_report.html"


def main():

    # Load conversation data
    df = pd.read_csv(CONVERSATION_PATH)

    print("Conversation data loaded:", df.shape)

    # Select columns useful for monitoring
    current_data = df[
        [
            "user_query",
            "intent",
            "feedback"
        ]
    ].copy()

    # For the first development version, use the same
    # dataset as both reference and current data.
    reference_data = current_data.copy()

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    result = report.run(
        current_data=current_data,
        reference_data=reference_data
    )

    result.save_html(REPORT_PATH)

    print("Evidently report created successfully!")
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()