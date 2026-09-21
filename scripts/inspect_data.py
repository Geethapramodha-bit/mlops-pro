import pandas as pd

DATA_PATH = "data/raw/intents.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:")
    print(df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nIntent distribution:")
    print(df["intent"].value_counts())


if __name__ == "__main__":
    main()