import os
import argparse
import joblib
import numpy as np
import pandas as pd

from utils import add_basic_text_features


def load_test_dataset(test_csv: str) -> pd.DataFrame:
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test file not found: {test_csv}")
    df = pd.read_csv(test_csv)
    expected_cols = {"sample_id", "catalog_content", "image_link"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in test data: {missing}")
    return df


def predict(model_path: str, test_csv: str, out_csv: str) -> None:
    model = joblib.load(model_path)
    df = load_test_dataset(test_csv)

    df = df.copy()
    df["catalog_content"] = df["catalog_content"].astype(str)
    df = add_basic_text_features(df, text_col="catalog_content")

    X_test = df[["catalog_content", "text_len", "num_digits", "ipq"]]

    y_log = model.predict(X_test)
    y_pred = np.expm1(np.maximum(y_log, 0.0))

    # enforce strictly positive
    y_pred = np.where(y_pred <= 0, 0.01, y_pred)

    out = pd.DataFrame({
        "sample_id": df["sample_id"],
        "price": y_pred.astype(float),
    })

    # Ensure the order matches input
    out = out.set_index("sample_id").reindex(df["sample_id"]).reset_index()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote predictions to {out_csv} with {len(out)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/text_ridge.joblib")
    parser.add_argument("--test_csv", default="dataset/test.csv")
    parser.add_argument("--out_csv", default="test_out.csv")
    args = parser.parse_args()
    predict(args.model_path, args.test_csv, args.out_csv)
