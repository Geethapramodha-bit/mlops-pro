
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


CONVERSATION_PATH = "data/conversations/conversations.csv"
REPORT_PATH = "reports/evidently_report.html"


def main():
    df = pd.read_csv(CONVERSATION_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    df = df.dropna(subset=["timestamp"])

    reference_data = df[
        df["timestamp"] < "2026-10-01"
    ][["user_query", "intent"]].copy()

    current_data = df[
        df["timestamp"] >= "2026-10-01"
    ][["user_query", "intent"]].copy()

    if reference_data.empty or current_data.empty:
        raise ValueError(
            "Both reference and current datasets must contain records."
        )

    print("Reference records:", len(reference_data))
    print("Current records:", len(current_data))

    report = Report(metrics=[DataDriftPreset()])

    result = report.run(
        current_data=current_data,
        reference_data=reference_data,
    )

    result.save_html(REPORT_PATH)

    print("Evidently drift report created successfully!")
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
