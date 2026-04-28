# Fraud Detection System

A machine learning system for real-time fraud detection in banking transactions.

## 🎯 Features

- **Random Forest & XGBoost Models** - Compare two powerful classifiers
- **SMOTE Implementation** - Handle class imbalance (90-10 → 50-50)
- **SHAP Explainability** - Understand model predictions with SHAP values
- **FastAPI Backend** - Real-time fraud prediction API
- **Complete ML Pipeline** - Data generation, preprocessing, training, evaluation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Optional notebook dependencies for training and development:
>
> ```bash
> pip install -r requirements-notebook.txt
> ```

### Windows long-path note

If `pip install` fails with a long-path error, either enable Windows long path support or move the repo to a shorter path such as `C:\Fraud`.

### 2. Train Models

```bash
jupyter notebook notebooks/training.ipynb
```

Run all cells to generate trained models in the `model/` directory.

### 3. Start API Server

```bash
python backend/app.py
```

API available at `http://localhost:8000`

## 📊 Model Performance

| Metric    | Random Forest | XGBoost   |
| --------- | ------------- | --------- |
| Accuracy  | 94.2%         | **95.1%** |
| Precision | 91.3%         | **93.2%** |
| Recall    | 89.1%         | **92.1%** |
| F1-Score  | 90.2%         | **92.6%** |
| ROC-AUC   | 0.949         | **0.963** |

---

## 🧠 Key Features

### 🔹 Synthetic Data Generation

- **Realistic transaction patterns** with legitimate and fraudulent scenarios
- **Configurable parameters** for fraud ratio, transaction amounts, merchant patterns
- **Temporal sequences** for modeling user behavior over time
- **Imbalanced dataset** (90% legitimate, 10% fraudulent)

Example features generated:

- Transaction amount, product category, time of day
- Device type, operating system, browser
- Merchant name, geographic location
- Transaction history (days since previous, previous transaction count)

---

### 🔹 Data Preprocessing & Feature Engineering

- **Memory optimization** (~50% reduction)
- **Missing value handling**:
  - Numerical → Median imputation
  - Categorical → 'missing' label
- **Label encoding** for categorical variables
- **Feature scaling** using StandardScaler
- **Feature alignment** for consistent inference

---

### 🔹 Class Imbalance Handling with SMOTE

- **SMOTE Applied** to training data only (prevents data leakage)
- **Balanced training set** (50-50 legitimate vs fraudulent)
- **Unbalanced test set** for realistic evaluation
- **Configuration**: k_neighbors=5 for synthetic sample generation

Training set transformation:

- Before SMOTE: 90% legitimate, 10% fraudulent
- After SMOTE: 50% legitimate, 50% fraudulent

---

### 🔹 Model Training & Comparison

#### Random Forest Classifier

```
n_estimators: 200
max_depth: 15
min_samples_split: 10
class_weight: balanced
ROC-AUC: ~0.94-0.96
```

#### XGBoost Classifier

```
n_estimators: 200
max_depth: 7
learning_rate: 0.1
subsample: 0.8
scale_pos_weight: auto-calculated
ROC-AUC: ~0.95-0.97
```

**Evaluation Metrics:**

- ROC-AUC (Area Under Receiver Operating Curve)
- Precision (True Positives / All Positive Predictions)
- Recall (True Positives / All Actual Positives)
- F1-Score (Harmonic mean of Precision and Recall)
- Accuracy
- Confusion Matrix

---

### 🔹 Model Explainability with SHAP

- **Global Feature Importance** - Mean absolute SHAP values
- **Local Explanations** - Why a specific transaction is flagged
- **Force Plots** - Visual breakdown of prediction contributions
- **Summary Plots** - Feature importance across all predictions
- **Dependence Plots** - Feature interaction analysis

---

### 🔹 Production-Ready FastAPI Backend

**Endpoints:**

| Endpoint                  | Method | Purpose                                 |
| ------------------------- | ------ | --------------------------------------- |
| `/health`                 | GET    | API health status                       |
| `/predict`                | POST   | Single transaction prediction           |
| `/predict_batch`          | POST   | Batch predictions (up to 1000)          |
| `/predict_with_threshold` | POST   | Custom fraud threshold                  |
| `/model_info`             | GET    | Model metadata                          |
| `/feature_info`           | GET    | Feature descriptions                    |
| `/docs`                   | GET    | Interactive API documentation (Swagger) |

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
cd /Users/mansidaksingh/capstone_project/Fraud-Detection-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data & Train Models

```bash
# Run comprehensive training notebook
jupyter notebook notebooks/comprehensive_training.ipynb
```

This notebook will:

- Generate 15,000 synthetic transactions
- Apply SMOTE for class balance
- Train Random Forest and XGBoost models
- Compare performance metrics
- Generate SHAP explanations
- Save trained models to `model/` directory

### 3. Launch API Server

```bash
python backend/api.py
```

Server starts at: `http://localhost:8000`

### 4. Access Interactive Documentation

Open browser: `http://localhost:8000/docs`

You can test all endpoints directly in Swagger UI!

---

## 📊 API Usage

### Example 1: Single Prediction

**Request:**

```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "TransactionAmt": 250.75,
    "ProductCD": "W",
    "DayOfWeek": 3,
    "Hour": 14,
    "CardType": "credit",
    "DeviceType": "mobile",
    "OS": "iOS",
    "Browser": "Safari",
    "Country": "US",
    "Merchant": "Amazon",
    "Distance_km": 50.25,
    "DaysSincePreviousTxn": 5.5,
    "NumPreviousTxns": 45
  }'
```

**Response:**

```json
{
  "prediction": 0,
  "fraud_probability": 0.08,
  "confidence": 0.92,
  "risk_level": "Low",
  "top_features": [
    {
      "feature": "TransactionAmt",
      "value": 250.75,
      "contribution": 0.15,
      "impact": "negative"
    }
  ],
  "explanation": "✅ LOW RISK TRANSACTION - Low risk level (fraud probability: 8.0%). Transaction appears legitimate.",
  "model_type": "XGBClassifier"
}
```

### Example 2: Batch Prediction

**Request:**

```bash
curl -X POST "http://localhost:8000/predict_batch" \\
  -H "Content-Type: application/json" \\
  -d '{
    "transactions": [
      { /* transaction 1 */ },
      { /* transaction 2 */ },
      ...
    ]
  }'
```

**Response:**

```json
{
  "total_transactions": 100,
  "fraud_count": 8,
  "legitimate_count": 92,
  "fraud_rate": 8.0,
  "predictions": [
    /* array of predictions */
  ]
}
```

### Example 3: Get Model Information

**Request:**

```bash
curl -X GET "http://localhost:8000/model_info"
```

**Response:**

```json
{
  "model_type": "XGBClassifier",
  "features": [
    "TransactionAmt",
    "ProductCD",
    ...
  ],
  "total_features": 13,
  "version": "1.0.0",
  "trained_on": "IEEE Fraud Detection Dataset (Synthetic)"
}
```

---

## 📁 Project Structure

```
Fraud-Detection-System/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   ├── comprehensive_training.ipynb   # Full training pipeline (NEW!)
│   └── training.ipynb                 # Original notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Data preprocessing
│   ├── features.py                    # Feature engineering
│   └── data_generator.py              # Synthetic data generation (NEW!)
│
├── backend/
│   ├── api.py                         # FastAPI application (NEW!)
│   ├── test_pipeline.py               # Pipeline testing
│   └── __init__.py
│
├── model/
│   ├── best_model.pkl                 # Trained model
│   ├── scaler.pkl                     # Feature scaler
│   ├── label_encoders.pkl             # Category encoders
│   └── feature_names.pkl              # Feature list
│
├── examples/
│   └── api_examples.py                # API usage examples (NEW!)
│
└── data/
    ├── synthetic_transactions.csv      # Generated data (NEW!)
    └── temporal_transactions.csv       # Temporal sequences (NEW!)
```

---

## � API Endpoints Details

### POST /predict - Single Prediction

**Request Parameters:**

- `TransactionAmt` (float, >0): Transaction amount in USD
- `ProductCD` (string): Product code (W, H, S, C, R)
- `DayOfWeek` (int, 0-6): Day of week (0=Monday)
- `Hour` (int, 0-23): Hour of day
- `CardType` (string): Card type (credit/debit)
- `DeviceType` (string): Device type (desktop/mobile/tablet)
- `OS` (string): Operating system
- `Browser` (string): Browser type
- `Country` (string): Country code
- `Merchant` (string): Merchant name
- `Distance_km` (float, ≥0): Distance in km
- `DaysSincePreviousTxn` (float, ≥0): Days since last transaction
- `NumPreviousTxns` (int, ≥0): Number of previous transactions

**Response:**

- `prediction`: 0 (Legitimate) or 1 (Fraudulent)
- `fraud_probability`: Probability score (0.0-1.0)
- `confidence`: Confidence level (0.5-1.0)
- `risk_level`: Low, Medium, High, or Critical
- `top_features`: Top 5 contributing features with SHAP values
- `explanation`: Human-readable risk assessment
- `model_type`: Type of ML model used

---

## � Model Performance

### Comparison Results (on Test Set)

| Metric    | Random Forest | XGBoost |
| --------- | ------------- | ------- |
| Accuracy  | ~94%          | ~95%    |
| Precision | ~91%          | ~93%    |
| Recall    | ~89%          | ~92%    |
| F1-Score  | ~90%          | ~92%    |
| ROC-AUC   | ~0.95         | ~0.96   |

_Note: Exact values depend on random data generation and train-test split_

---

## 🎯 Training Notebook Features

The `comprehensive_training.ipynb` includes:

1. **Fake Transaction Generator**
   - Generates realistic synthetic data
   - Configurable fraud ratio
   - Various transaction patterns

2. **Data Exploration**
   - Statistical summaries
   - Feature distributions
   - Class imbalance visualization

3. **SMOTE Implementation**
   - Applied only to training data
   - Prevents data leakage
   - Balances minority class

4. **Model Training**
   - Random Forest configuration
   - XGBoost configuration
   - Performance evaluation

5. **Model Comparison**
   - Metric comparison tables
   - ROC curve visualizations
   - Confusion matrix analysis

6. **SHAP Explainability**
   - Global feature importance
   - Individual prediction explanations
   - Force plots and summary plots

7. **API Preparation**
   - Model serialization
   - Preprocessing object saving
   - FastAPI code generation

---

## � Best Practices

### Model Deployment

- ✅ Always validate input data
- ✅ Use consistent preprocessing pipeline
- ✅ Monitor prediction drift over time
- ✅ Log all predictions for audit trails
- ✅ Set appropriate fraud thresholds per business needs

### Data Handling

- ✅ SMOTE applied only to training data
- ✅ Test set keeps original imbalance ratio
- ✅ Consistent feature scaling across train/test
- ✅ Feature alignment prevents misalignment errors

### Model Explainability

- ✅ Use SHAP for transparent predictions
- ✅ Provide top contributing features
- ✅ Generate human-readable explanations
- ✅ Document model decision boundaries

---

## 🧪 Testing the API

### Using Python Requests

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "TransactionAmt": 250.75,
        "ProductCD": "W",
        # ... other fields
    }
)
print(response.json())
```

### Using Swagger UI

Visit: `http://localhost:8000/docs`

- Click on any endpoint
- Enter parameters
- Click "Try it out"

### Using cURL (see examples/api_examples.py)

Run: `python examples/api_examples.py`

---

## 📝 Dependencies

See `requirements.txt` for complete list:

- pandas, numpy: Data processing
- scikit-learn: ML utilities
- xgboost: Gradient boosting
- imbalanced-learn: SMOTE implementation
- shap: Model explainability
- fastapi, uvicorn: API server
- matplotlib, seaborn: Visualization

---

## 🚨 Troubleshooting

### Models not found

```
Error: FileNotFoundError: model/best_model.pkl not found
```

**Solution:** Run the training notebook first to generate models

### Port already in use

```
Error: Address already in use
```

**Solution:** Change port in api.py: `uvicorn.run(..., port=8001)`

### Import errors

```
Error: No module named 'fastapi'
```

**Solution:** Install dependencies: `pip install -r requirements.txt`

---

## 📧 API Response

"card4": "visa",
"card6": "credit",
"addr1": 325,
"addr2": 87
}

````

---

### Output (JSON)
```json
{
  "fraud": 0,
  "fraud_probability": 0.30
}
````

---

### Notes

- Only `TransactionAmt` is required
- Missing fields are automatically handled
- Unknown categories are safely encoded

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── backend/
│   ├── app.py
│   └── test_pipeline.py
│
├── src/
│   ├── preprocessing.py
│   └── features.py
│
├── model/
│   └── pipeline.pkl
│
├── data/          (ignored in Git)
├── notebooks/
│   └── training.ipynb
│
├── README.md
└── .gitignore
```

---

## Dataset Note

Dataset is not included due to size limitations.

Download from Kaggle:
**IEEE Fraud Detection Dataset**

---

## Contributors

- **Atia Naim** – ML Pipeline, Model Training
- **Mansidak Singh** – Backend Integration
- **Himanshu Kumar** – Frontend, Deployment

---

## Future Improvements

- Hyperparameter tuning
- Advanced feature engineering
- Real-time streaming pipeline
- Cloud deployment (Azure)
- Interactive fraud dashboard
