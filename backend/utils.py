"""
Utility functions for preprocessing and model management.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path


class ModelManager:
    """Manage model loading and predictions."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.explainer = None
    
    def load_models(self):
        """Load all models and preprocessors."""
        try:
            with open(self.model_dir / "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(self.model_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(self.model_dir / "encoder.pkl", "rb") as f:
                self.label_encoders = pickle.load(f)
            with open(self.model_dir / "features.json", "r") as f:
                self.feature_names = json.load(f)
            
            # Initialize SHAP explainer
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def preprocess(self, data_dict: dict) -> np.ndarray:
        """Preprocess transaction data."""
        df = pd.DataFrame([data_dict])
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                df[col] = encoder.transform([val])[0] if val in encoder.classes_ else -1
        
        # Reindex to match training features
        df = df.reindex(columns=self.feature_names, fill_value=0)
        
        # Scale features
        return self.scaler.transform(df)
    
    def predict(self, X: np.ndarray) -> tuple:
        """Make prediction and get probabilities."""
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        fraud_probability = probabilities[1]
        confidence = max(probabilities)
        
        return prediction, fraud_probability, confidence
    
    def explain(self, X: np.ndarray, top_k: int = 5) -> list:
        """Get SHAP explanations."""
        shap_values = self.explainer.shap_values(X)
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap': np.abs(shap_values[0])
        }).sort_values('shap', ascending=False).head(top_k)
        
        return [
            {
                "feature": row['feature'],
                "importance": float(row['shap'])
            }
            for _, row in importance.iterrows()
        ]


def get_risk_level(fraud_probability: float) -> str:
    """Determine risk level based on fraud probability."""
    if fraud_probability < 0.3:
        return "Low"
    elif fraud_probability < 0.6:
        return "Medium"
    elif fraud_probability < 0.8:
        return "High"
    else:
        return "Critical"
