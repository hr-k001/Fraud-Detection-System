import json
import pickle
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import DataPreprocessor  # noqa: F401,E402
from src.features import FeatureEngineer  # noqa: F401,E402


def load_pipeline(path=BASE_DIR / "model" / "pipeline.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_input(data, pipeline):
    df = pd.DataFrame([data])
    df = pipeline["preprocessor"].transform(df)
    df = pipeline["feature_engineer"].transform(df)
    df = df.reindex(columns=pipeline["features"], fill_value=0)
    return df


def predict(data, pipeline):
    df = preprocess_input(data, pipeline)
    model = pipeline["model"]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return pred, prob


if __name__ == "__main__":
    pipeline = load_pipeline()

    with open(Path(__file__).with_name("test_cases.json"), "r") as f:
        test_cases = json.load(f)

    print("\nRunning Test Cases...\n")

    for i, sample in enumerate(test_cases, 1):
        pred, prob = predict(sample, pipeline)

        print(f"Test Case {i}")
        print("Input:", sample)
        print("Prediction:", pred)
        print("Fraud Probability:", round(prob, 4))
        print("-" * 50)
