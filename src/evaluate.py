import pandas as pd

TEST_PATH = "data/processed/test_predictions.csv"
VAL_PATH = "data/processed/val_predictions.csv"

def load_predictions():
    test_df = pd.read_csv(TEST_PATH)
    val_df = pd.read_csv(VAL_PATH)

    print("Test shape:", test_df.shape)
    print("Validation shape:", val_df.shape)

    print("\nColumns:", test_df.columns.tolist())

    return test_df, val_df


if __name__ == "__main__":
    print("=== Loading Predictions ===\n")
    test_df, val_df = load_predictions()
    print("\nDone!")