import os
import json
import joblib
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from utils import add_basic_text_features, smape


def load_dataset(train_csv: str) -> pd.DataFrame:
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Training file not found: {train_csv}")
    df = pd.read_csv(train_csv)
    expected_cols = {"sample_id", "catalog_content", "image_link", "price"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")
    return df


def build_pipeline(max_features: int = 200000, alpha: float = 2.0) -> Pipeline:
    text_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        strip_accents="unicode",
        lowercase=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", text_vectorizer, "catalog_content"),
            ("num", StandardScaler(with_mean=False), ["text_len", "num_digits", "ipq"]),
        ],
        sparse_threshold=0.3,
        remainder="drop",
    )

    model = Ridge(alpha=alpha, random_state=42)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    return pipeline


def train(train_csv: str, model_out: str, val_size: float = 0.1, random_state: int = 42) -> None:
    df = load_dataset(train_csv)

    # Basic cleanup
    df = df.dropna(subset=["catalog_content", "price"]).copy()
    df["catalog_content"] = df["catalog_content"].astype(str)

    # Feature engineering
    df = add_basic_text_features(df, text_col="catalog_content")

    # Target transform: log1p for stability
    y = np.log1p(df["price"].values.astype(float))

    X = df[["catalog_content", "text_len", "num_digits", "ipq"]]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Validation metrics
    y_val_pred = pipeline.predict(X_val)
    y_val_pred_price = np.expm1(np.maximum(y_val_pred, 0.0))
    y_val_price = np.expm1(y_val)

    mae = mean_absolute_error(y_val_price, y_val_pred_price)
    metric_smape = smape(y_val_price, y_val_pred_price)

    print(json.dumps({
        "val_mae": mae,
        "val_smape": metric_smape,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
    }, indent=2))

    # Persist model
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"Saved model to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="dataset/train.csv")
    parser.add_argument("--model_out", default="models/text_ridge.joblib")
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()
    train(args.train_csv, args.model_out, val_size=args.val_size)
