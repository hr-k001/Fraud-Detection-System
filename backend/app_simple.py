"""
FastAPI Application for Fraud Detection System (Simplified - No SHAP)

Real-time fraud prediction with feature importance.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with XGBoost",
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


def load_models():
    """Load trained models and preprocessing objects."""
    global model, scaler, label_encoders, feature_names
    
    try:
        with open(MODEL_DIR / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / "label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        with open(MODEL_DIR / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        
        print("✅ Models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False


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


def preprocess_transaction(transaction_dict: dict) -> np.ndarray:
    """Preprocess transaction data for prediction."""
    try:
        # Create DataFrame from transaction
        df = pd.DataFrame([transaction_dict])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                if val in encoder.classes_:
                    df[col] = encoder.transform([val])[0]
                else:
                    df[col] = -1  # Unknown category
        
        # Reindex to match training features
        df = df.reindex(columns=feature_names, fill_value=0)
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        return X_scaled
    except Exception as e:
        raise ValueError(f"Error preprocessing transaction: {e}")


# Request/Response Models
class Transaction(BaseModel):
    """Single transaction for prediction."""
    TransactionAmt: float = Field(..., gt=0, le=100000, description="Transaction amount")
    ProductCD: str = Field(..., description="Product code")
    DayOfWeek: int = Field(..., ge=0, le=6, description="Day of week (0-6)")
    Hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    CardType: str = Field(..., description="Card type (credit/debit)")
    DeviceType: str = Field(..., description="Device type (desktop/mobile/tablet)")
    OS: str = Field(..., description="Operating system")
    Browser: str = Field(..., description="Browser type")
    Country: str = Field(..., description="Country code")
    Merchant: str = Field(..., description="Merchant name")
    Distance_km: float = Field(..., ge=0, description="Distance in km")
    DaysSincePreviousTxn: float = Field(..., ge=0, description="Days since previous transaction")
    NumPreviousTxns: int = Field(..., ge=0, description="Number of previous transactions")


class FeatureImportance(BaseModel):
    """Feature importance information."""
    feature: str
    importance: float


class PredictionResponse(BaseModel):
    """API response for prediction."""
    prediction: int = Field(..., description="0: Legitimate, 1: Fraudulent")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    confidence: float = Field(..., description="Confidence in prediction (0-1)")
    risk_level: str = Field(..., description="Risk level: Low/Medium/High/Critical")
    top_features: List[FeatureImportance] = Field(..., description="Top 5 contributing features")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    transactions: List[Transaction] = Field(..., description="List of transactions")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    total_transactions: int
    fraud_count: int
    legitimate_count: int
    fraud_rate: float
    predictions: List[PredictionResponse]


class ModelInfo(BaseModel):
    """Model information."""
    model_type: str
    features: List[str]
    total_features: int
    version: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("\n" + "=" * 70)
    print("🚀 FRAUD DETECTION API - STARTUP")
    print("=" * 70)
    load_models()
    print("📊 API Ready for predictions!")
    print("=" * 70 + "\n")


# Health Check Endpoint
@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Fraud Detection API",
        "model_loaded": model is not None
    }


# Single Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction) -> PredictionResponse:
    """
    Predict if a single transaction is fraudulent.
    
    Args:
        transaction: Transaction details
    
    Returns:
        PredictionResponse with prediction, probability, and risk level
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess transaction
        X_scaled = preprocess_transaction(transaction.dict())
        
        # Make prediction
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]
        fraud_probability = float(probabilities[1])
        confidence = float(max(probabilities))
        
        # Determine risk level
        risk_level = get_risk_level(fraud_probability)
        
        # Get feature importance (from model if available)
        top_features = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(5)
            
            top_features = [
                FeatureImportance(
                    feature=row['feature'],
                    importance=float(row['importance'])
                )
                for _, row in feature_imp.iterrows()
            ]
        
        return PredictionResponse(
            prediction=prediction,
            fraud_probability=fraud_probability,
            confidence=confidence,
            risk_level=risk_level,
            top_features=top_features
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Batch Prediction Endpoint
@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict fraud for multiple transactions.
    
    Args:
        request: BatchPredictionRequest with list of transactions
    
    Returns:
        Batch results with aggregate statistics
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        fraud_count = 0
        
        for transaction in request.transactions:
            result = await predict_fraud(transaction)
            predictions.append(result)
            if result.prediction == 1:
                fraud_count += 1
        
        total = len(predictions)
        legitimate_count = total - fraud_count
        fraud_rate = (fraud_count / total * 100) if total > 0 else 0.0
        
        return BatchPredictionResponse(
            total_transactions=total,
            fraud_count=fraud_count,
            legitimate_count=legitimate_count,
            fraud_rate=fraud_rate,
            predictions=predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# Model Info Endpoint
@app.get("/model_info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Get information about the model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type=type(model).__name__,
        features=feature_names,
        total_features=len(feature_names),
        version="1.0.0"
    )


# Root Endpoint
@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "description": "Real-time fraud prediction using XGBoost",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "model_info": "/model_info",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
