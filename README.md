#  Fraud Detection System (Real-Time Banking Transactions)

##  Overview
This project implements a **real-time fraud detection system** using machine learning on the IEEE Fraud Detection dataset.

The system processes transaction data, predicts fraudulent activity, and exposes results through a **FastAPI backend**, making it ready for integration with a frontend dashboard.

---

##  Objectives
- Detect fraudulent transactions in near real-time  
- Handle highly imbalanced data  
- Build a scalable ML pipeline  
- Provide explainability for predictions  

---

##  Tech Stack

| Component | Technology |
|----------|-----------|
| Language | Python |
| ML Model | XGBoost |
| Backend | FastAPI |
| Data Processing | Pandas, NumPy |
| Imbalance Handling | SMOTE |
| Explainability | SHAP |
| Version Control | Git, GitHub |

---

## 🧠 Key Features

### 🔹 Data Preprocessing
- Memory optimization (~50% reduction)
- Missing value handling:
  - Numerical → Median
  - Categorical → "missing"
- Dropped columns with >80% missing values

---

### 🔹 Feature Engineering
- Label Encoding for categorical variables  
- Feature alignment for consistent inference  

---

### 🔹 Model Training
- Model: **XGBoost Classifier**
- Imbalance handling:
  - `scale_pos_weight`
  - SMOTE 
- Evaluation Metrics:
  - ROC-AUC  
  - Precision / Recall  
  - F1 Score  
  - PR-AUC  

---

### 🔹 Model Explainability
- Implemented SHAP for:
  - Global feature importance  
  - Local prediction explanations  

---

### 🔹 Pipeline
Reusable pipeline including:
- Preprocessing  
- Encoding  
- Model  

Saved as:
```
model/pipeline.pkl
```

---

### 🔹 Backend API
Built using **FastAPI** for real-time predictions.

---

## 🔌 API Usage

### Endpoint
```
POST /predict
```

---

### Input (JSON)
```json
{
  "TransactionAmt": 250.75,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "credit",
  "addr1": 325,
  "addr2": 87
}
```

---

### Output (JSON)
```json
{
  "fraud": 0,
  "fraud_probability": 0.30
}
```

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

##  Dataset Note
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


