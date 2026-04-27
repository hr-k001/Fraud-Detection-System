import pickle
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..'))

from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer



# load pipeline
with open("../model/pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# sample input
sample = {
    "TransactionAmt": 100,
    "ProductCD": "W"
}

df = pd.DataFrame([sample])

# pipeline steps
df = pipeline["preprocessor"].transform(df)
df = pipeline["feature_engineer"].transform(df)

df = df.reindex(columns=pipeline["features"], fill_value=0)

# predict
model = pipeline["model"]

pred = model.predict(df)[0]
prob = model.predict_proba(df)[0][1]

print("Prediction:", pred)
print("Fraud Probability:", prob)