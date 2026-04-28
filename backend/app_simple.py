"""
FastAPI application for the fraud detection system.

This app loads the committed model/pipeline.pkl artifact, which already
contains the trained model, preprocessing object, feature engineering object,
and final feature list from notebooks/training.ipynb.
"""

from pathlib import Path
from typing import List
import pickle
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
sys.path.insert(0, str(BASE_DIR))

from backend.database import PredictionStore
from src.features import FeatureEngineer
from src.preprocessing import DataPreprocessor


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with the saved preprocessing pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pipeline = None
model = None
preprocessor = None
feature_engineer = None
feature_names = None
prediction_store = PredictionStore()


class PipelineUnpickler(pickle.Unpickler):
    """Map notebook-defined classes back to source modules while loading."""

    def find_class(self, module, name):
        if module == "__main__" and name == "DataPreprocessor":
            return DataPreprocessor
        if module == "__main__" and name == "FeatureEngineer":
            return FeatureEngineer
        return super().find_class(module, name)


def load_models() -> bool:
    """Load the saved inference pipeline."""
    global pipeline, model, preprocessor, feature_engineer, feature_names

    try:
        with open(PIPELINE_PATH, "rb") as f:
            pipeline = PipelineUnpickler(f).load()

        model = pipeline["model"]
        preprocessor = pipeline["preprocessor"]
        feature_engineer = pipeline["feature_engineer"]
        feature_names = pipeline["features"]

        print(f"Models loaded successfully from {PIPELINE_PATH}")
        return True
    except Exception as exc:
        print(f"Error loading models: {exc}")
        return False


def get_risk_level(fraud_probability: float) -> str:
    """Determine risk level based on fraud probability."""
    if fraud_probability < 0.3:
        return "Low"
    if fraud_probability < 0.6:
        return "Medium"
    if fraud_probability < 0.8:
        return "High"
    return "Critical"


def preprocess_transaction(transaction_dict: dict) -> pd.DataFrame:
    """Preprocess transaction data using the saved training pipeline objects."""
    try:
        df = pd.DataFrame([transaction_dict])
        df = preprocessor.transform(df)
        df = feature_engineer.transform(df)
        df = df.reindex(columns=feature_names, fill_value=0)
        return df
    except Exception as exc:
        raise ValueError(f"Error preprocessing transaction: {exc}") from exc


class Transaction(BaseModel):
    """Single transaction for prediction.

    The IEEE model has many optional fields. TransactionAmt and ProductCD are
    kept explicit, while extra fields such as card1, card4, addr1, C1, etc.
    are accepted and passed into the saved preprocessing pipeline.
    """

    model_config = ConfigDict(extra="allow")

    TransactionAmt: float = Field(..., gt=0, description="Transaction amount")
    ProductCD: str = Field(..., description="Product code")


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

    transactions: List[Transaction] = Field(
        ..., max_length=1000, description="List of transactions"
    )


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
    artifact: str


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("\n" + "=" * 70)
    print("FRAUD DETECTION API - STARTUP")
    print("=" * 70)
    load_models()
    prediction_store.init_schema()
    print("API ready for predictions")
    print("=" * 70 + "\n")


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Fraud Detection API",
        "model_loaded": model is not None,
        "artifact": str(PIPELINE_PATH.relative_to(BASE_DIR)),
    }


@app.get("/storage/status")
async def storage_status() -> dict:
    """Return Azure SQL storage status."""
    return prediction_store.status()


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction) -> PredictionResponse:
    """Predict if a single transaction is fraudulent."""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        X = preprocess_transaction(transaction.model_dump())

        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        fraud_probability = float(probabilities[1])
        confidence = float(max(probabilities))
        risk_level = get_risk_level(fraud_probability)

        top_features = []
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False).head(5)

            top_features = [
                FeatureImportance(
                    feature=row["feature"],
                    importance=float(row["importance"]),
                )
                for _, row in feature_imp.iterrows()
            ]

        response = PredictionResponse(
            prediction=prediction,
            fraud_probability=fraud_probability,
            confidence=confidence,
            risk_level=risk_level,
            top_features=top_features,
        )
        prediction_store.save_prediction(transaction.model_dump(), response.model_dump())
        return response

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict fraud for multiple transactions."""
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
        fraud_rate = (fraud_count / total * 100) if total else 0.0

        return BatchPredictionResponse(
            total_transactions=total,
            fraud_count=fraud_count,
            legitimate_count=legitimate_count,
            fraud_rate=fraud_rate,
            predictions=predictions,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc


@app.get("/model_info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_type=type(model).__name__,
        features=feature_names,
        total_features=len(feature_names),
        version="1.0.0",
        artifact=str(PIPELINE_PATH.relative_to(BASE_DIR)),
    )


@app.get("/transactions")
async def recent_transactions(limit: int = 50) -> dict:
    """Return recent checked transactions from Azure SQL or memory fallback."""
    rows = prediction_store.recent_predictions(limit=limit, only_alerts=False)
    return {"total": len(rows), "transactions": rows}


@app.get("/alerts")
async def recent_alerts(limit: int = 50) -> dict:
    """Return recent fraud/high-risk alerts."""
    rows = prediction_store.recent_predictions(limit=limit, only_alerts=True)
    return {"total": len(rows), "alerts": rows}


from fastapi.responses import FileResponse

frontend_dist = BASE_DIR / "frontend" / "dist"

# Serve React build
if frontend_dist.exists():
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(frontend_dist / "index.html")

    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
