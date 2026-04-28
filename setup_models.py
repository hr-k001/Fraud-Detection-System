"""
Quick setup script to generate and save trained models
"""
import sys
sys.path.insert(0, '/Users/mansidaksingh/capstone_project/Fraud-Detection-System')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import pickle
import os

# Import the data generator
from src.data_generator import FakeTransactionGenerator

print("=" * 60)
print("SETTING UP MODELS FOR FASTAPI")
print("=" * 60)

# Generate synthetic data
print("\n1. Generating synthetic transaction data...")
generator = FakeTransactionGenerator(seed=42)
df = generator.generate_transactions(n_samples=10000, fraud_ratio=0.10)
print(f"   ✓ Generated {len(df):,} transactions")

# Preprocess data
print("\n2. Preprocessing data...")
df_processed = df.copy()
df_processed = df_processed.drop(['TransactionID'], axis=1)

categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('isFraud')

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le

print(f"   ✓ Encoded {len(categorical_cols)} categorical columns")

# Split data
X = df_processed.drop('isFraud', axis=1)
y = df_processed['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train/Test split: {len(X_train)} / {len(X_test)}")

# Apply SMOTE
print("\n3. Applying SMOTE for class imbalance...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"   ✓ After SMOTE: {len(X_train_smote)} samples (balanced)")

# Scale features
print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)
print(f"   ✓ Features scaled")

# Train XGBoost model
print("\n5. Training XGBoost model...")
scale_pos_weight = (y_train_smote == 0).sum() / (y_train_smote == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    verbosity=0
)
xgb_model.fit(X_train_scaled, y_train_smote)
print(f"   ✓ Model trained successfully")

# Evaluate model
from sklearn.metrics import accuracy_score, roc_auc_score
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, xgb_pred)
roc_auc = roc_auc_score(y_test, xgb_pred_proba)
print(f"   ✓ Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

# Save models
print("\n6. Saving models...")
model_dir = "/Users/mansidaksingh/capstone_project/Fraud-Detection-System/model"
os.makedirs(model_dir, exist_ok=True)

with open(f'{model_dir}/best_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"   ✓ Saved best_model.pkl")

with open(f'{model_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Saved scaler.pkl")

with open(f'{model_dir}/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"   ✓ Saved label_encoders.pkl")

feature_names = X.columns.tolist()
with open(f'{model_dir}/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print(f"   ✓ Saved feature_names.pkl")

print("\n" + "=" * 60)
print("✓ SETUP COMPLETE!")
print("=" * 60)
print(f"\nModels saved in: {model_dir}/")
print("Ready to run FastAPI backend!")
