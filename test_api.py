"""
Test script for Fraud Detection API
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

print("=" * 70)
print("FRAUD DETECTION API TEST SUITE")
print("=" * 70)

# Test 1: Health Check
print("\n1️⃣  TEST: Health Check")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Single Prediction - Legitimate Transaction
print("\n2️⃣  TEST: Single Prediction - Legitimate Transaction")
print("-" * 70)
legitimate_txn = {
    "TransactionAmt": 150.00,
    "ProductCD": "W",
    "DayOfWeek": 3,
    "Hour": 14,
    "CardType": "credit",
    "DeviceType": "desktop",
    "OS": "Windows",
    "Browser": "Chrome",
    "Country": "US",
    "Merchant": "Amazon",
    "Distance_km": 50.0,
    "DaysSincePreviousTxn": 10.5,
    "NumPreviousTxns": 45
}

try:
    response = requests.post(
        f"{BASE_URL}/predict",
        json=legitimate_txn
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nPrediction Result:")
    print(f"  • Fraud (0=No, 1=Yes): {result['prediction']}")
    print(f"  • Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.2f}%)")
    print(f"  • Confidence: {result['confidence']:.4f}")
    print(f"  • Risk Level: {result.get('risk_level', 'N/A')}")
    if 'top_features' in result:
        print(f"\n  Top Contributing Features:")
        for feature in result['top_features'][:5]:
            print(f"    - {feature['feature']}: {feature.get('importance', 'N/A')}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Single Prediction - Suspicious Transaction
print("\n3️⃣  TEST: Single Prediction - Suspicious Transaction")
print("-" * 70)
suspicious_txn = {
    "TransactionAmt": 3500.00,
    "ProductCD": "W",
    "DayOfWeek": 2,
    "Hour": 2,                # 2 AM - unusual
    "CardType": "credit",
    "DeviceType": "mobile",
    "OS": "iOS",
    "Browser": "Safari",
    "Country": "GB",
    "Merchant": "Unknown",
    "Distance_km": 2000.0,    # Very far
    "DaysSincePreviousTxn": 0.1,  # Rapid transaction
    "NumPreviousTxns": 1      # New account
}

try:
    response = requests.post(
        f"{BASE_URL}/predict",
        json=suspicious_txn
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nPrediction Result:")
    print(f"  • Fraud (0=No, 1=Yes): {result['prediction']}")
    print(f"  • Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.2f}%)")
    print(f"  • Confidence: {result['confidence']:.4f}")
    print(f"  • Risk Level: {result.get('risk_level', 'N/A')}")
    if 'top_features' in result:
        print(f"\n  Top Contributing Features:")
        for feature in result['top_features'][:5]:
            print(f"    - {feature['feature']}: {feature.get('importance', 'N/A')}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Batch Prediction
print("\n4️⃣  TEST: Batch Prediction (3 transactions)")
print("-" * 70)
batch_txns = {
    "transactions": [
        legitimate_txn,
        suspicious_txn,
        {
            "TransactionAmt": 99.99,
            "ProductCD": "S",
            "DayOfWeek": 5,
            "Hour": 18,
            "CardType": "debit",
            "DeviceType": "mobile",
            "OS": "Android",
            "Browser": "Chrome",
            "Country": "US",
            "Merchant": "Walmart",
            "Distance_km": 30.0,
            "DaysSincePreviousTxn": 2.0,
            "NumPreviousTxns": 102
        }
    ]
}

try:
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=batch_txns
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nBatch Processing Result:")
    print(f"  • Total Transactions: {result['total_transactions']}")
    print(f"  • Fraud Count: {result['fraud_count']}")
    print(f"  • Legitimate Count: {result['total_transactions'] - result['fraud_count']}")
    print(f"  • Fraud Rate: {(result['fraud_count'] / result['total_transactions'] * 100):.1f}%")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Model Information
print("\n5️⃣  TEST: Model Information")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nModel Information:")
    print(f"  • Model Type: {result['model_type']}")
    print(f"  • Total Features: {result['total_features']}")
    print(f"  • API Version: {result['version']}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 6: Interactive Swagger UI
print("\n6️⃣  INFO: Interactive Testing")
print("-" * 70)
print(f"Swagger UI: http://127.0.0.1:8000/docs")
print(f"ReDoc: http://127.0.0.1:8000/redoc")

print("\n" + "=" * 70)
print("✅ TEST SUITE COMPLETE!")
print("=" * 70)
