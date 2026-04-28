"""
FastAPI Application for Fraud Detection System

Real-time fraud prediction with SHAP explanations.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with SHAP explanations",
    version="1.0.0"
)

# Model directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

# Global variables
model = None
scaler = None
label_encoders = None
feature_names = None
explainer = None


def load_models():
    """Load trained models and preprocessing objects."""
    global model, scaler, label_encoders, feature_names, explainer
    
    try:
        with open(MODEL_DIR / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / "label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        with open(MODEL_DIR / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        
        explainer = shap.TreeExplainer(model)
        print("✓ Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error loading models: {e}")


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


# ==================== Request/Response Models ====================

class Transaction(BaseModel):
    """Single transaction for prediction."""
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount")
    ProductCD: str = Field(..., description="Product code")
    DayOfWeek: int = Field(..., ge=0, le=6, description="Day of week")
    Hour: int = Field(..., ge=0, le=23, description="Hour of day")
    CardType: str = Field(..., description="Card type")
    DeviceType: str = Field(..., description="Device type")
    OS: str = Field(..., description="Operating system")
    Browser: str = Field(..., description="Browser type")
    Country: str = Field(..., description="Country code")
    Merchant: str = Field(..., description="Merchant name")
    Distance_km: float = Field(..., ge=0, description="Distance in km")
    DaysSincePreviousTxn: float = Field(..., ge=0, description="Days since previous")
    NumPreviousTxns: int = Field(..., ge=0, description="Previous transactions")
    
    @validator('TransactionAmt')
    def validate_amount(cls, v):
        if v > 100000:
            raise ValueError('Amount too high')
        return v


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: int = Field(..., description="0=Legitimate, 1=Fraudulent")
    fraud_probability: float = Field(..., description="Fraud probability")
    confidence: float = Field(..., description="Confidence score")
    risk_level: str = Field(..., description="Risk level")
    top_features: List[Dict[str, Any]] = Field(..., description="Top features")


class BatchRequest(BaseModel):
    """Batch prediction request."""
    transactions: List[Transaction] = Field(..., max_items=1000)


# ==================== Helper Functions ====================

def preprocess_transaction(txn_dict: Dict) -> np.ndarray:
    """Preprocess transaction data."""
    df = pd.DataFrame([txn_dict])
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val in encoder.classes_:
                df[col] = encoder.transform([val])[0]
            else:
                df[col] = -1
    
    # Ensure correct feature order
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # Scale
    return scaler.transform(df)


def get_risk_level(prob: float) -> str:
    """Determine risk level."""
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    elif prob < 0.8:
        return "High"
    else:
        return "Critical"


# ==================== API Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """Predict fraud for single transaction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess
        X = preprocess_transaction(transaction.dict())
        
        # Predict
        prediction = model.predict(X)[0]
        fraud_prob = model.predict_proba(X)[0][1]
        confidence = max(model.predict_proba(X)[0])
        
        # Get SHAP explanation
        shap_values = explainer.shap_values(X)
        importance = pd.DataFrame({
            'feature': feature_names,
            'shap': np.abs(shap_values[0])
        }).sort_values('shap', ascending=False).head(5)
        
        top_features = [
            {
                "feature": row['feature'],
                "importance": float(row['shap'])
            }
            for _, row in importance.iterrows()
        ]
        
        return PredictionResponse(
            prediction=int(prediction),
            fraud_probability=float(fraud_prob),
            confidence=float(confidence),
            risk_level=get_risk_level(fraud_prob),
            top_features=top_features
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(request: BatchRequest):
    """Batch predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        fraud_count = 0
        
        for txn in request.transactions:
            pred = await predict_fraud(txn)
            results.append(pred.dict())
            if pred.prediction == 1:
                fraud_count += 1
        
        return {
            "total": len(results),
            "fraud_count": fraud_count,
            "fraud_rate": fraud_count / len(results) * 100,
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/docs")
async def get_docs():
    """API documentation."""
    return {"message": "Use /docs (Swagger UI) or /redoc (ReDoc)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
