from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "train.csv"
OUTPUT_DIR = BASE_DIR / "data" / "splits"


def main():
    df = pd.read_csv(CSV_PATH)

    required = {"id_code", "diagnosis"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # First split:
    # 85% temporary training/validation
    # 15% final test
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df["diagnosis"],
    )

    # Second split:
    # From the remaining 85%:
    # 70% overall train
    # 15% overall validation
    val_ratio_inside_train_val = 0.15 / 0.85

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_inside_train_val,
        random_state=RANDOM_STATE,
        stratify=train_val_df["diagnosis"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "val.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("RETINOXAI DATA SPLIT")
    print("====================")
    print(f"Total      : {len(df)}")
    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")

    print("\nTrain distribution:")
    print(train_df["diagnosis"].value_counts().sort_index())

    print("\nValidation distribution:")
    print(val_df["diagnosis"].value_counts().sort_index())

    print("\nTest distribution:")
    print(test_df["diagnosis"].value_counts().sort_index())

    print("\nSaved:")
    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()