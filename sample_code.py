"""
Minimal baseline to train a text-only model and generate predictions.
Usage:
  python sample_code.py --train dataset/train.csv --test dataset/test.csv --out test_out.csv
"""
import os
import argparse

from src.train import train as train_fn
from src.predict import predict as predict_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--model", default="models/text_ridge.joblib")
    parser.add_argument("--out", default="test_out.csv")
    args = parser.parse_args()

    # Train
    train_fn(train_csv=args.train, model_out=args.model)

    # Predict
    predict_fn(model_path=args.model, test_csv=args.test, out_csv=args.out)


if __name__ == "__main__":
    main()
