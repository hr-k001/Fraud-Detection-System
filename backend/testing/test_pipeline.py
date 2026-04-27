import pickle
import pandas as pd
import sys
import os
import json

# Fix import path (so src/ works)
sys.path.append(os.path.abspath('../..'))

from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer


# ---------------------------
# LOAD PIPELINE
# ---------------------------
def load_pipeline(path="../../model/pipeline.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------
# PREPROCESS INPUT
# ---------------------------
def preprocess_input(data, pipeline):
    df = pd.DataFrame([data])

    df = pipeline["preprocessor"].transform(df)
    df = pipeline["feature_engineer"].transform(df)

    df = df.reindex(columns=pipeline["features"], fill_value=0)

    return df


# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict(data, pipeline):
    df = preprocess_input(data, pipeline)

    model = pipeline["model"]

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return pred, prob


# ---------------------------
# MAIN TESTING
# ---------------------------
if __name__ == "__main__":
    
    pipeline = load_pipeline()

    # Load test cases from JSON
    with open("test_cases.json", "r") as f:
        test_cases = json.load(f)

    print("\n🚀 Running Test Cases...\n")

    for i, sample in enumerate(test_cases, 1):
        pred, prob = predict(sample, pipeline)

        print(f"Test Case {i}")
        print("Input:", sample)
        print("Prediction:", pred)
        print("Fraud Probability:", round(prob, 4))
        print("-" * 50)