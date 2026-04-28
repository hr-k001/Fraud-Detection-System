"""Storage support for fraud prediction history.

The app stores predictions in Azure SQL when the connection is available and
keeps a small in-memory mirror so the dashboard remains usable during local
development or firewall interruptions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import threading


BASE_DIR = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """Load simple KEY=VALUE pairs without requiring an extra dependency."""
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class PredictionStore:
    """Store and retrieve prediction history."""

    def __init__(self, max_memory_rows: int = 500):
        load_dotenv()
        self.max_memory_rows = max_memory_rows
        self.memory_rows: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.last_error: Optional[str] = None
        self.pyodbc = None
        self.connection_string = self._build_connection_string()

        try:
            import pyodbc  # type: ignore

            self.pyodbc = pyodbc
        except Exception as exc:
            self.last_error = f"pyodbc not installed or unavailable: {exc}"

    def _build_connection_string(self) -> Optional[str]:
        server = os.getenv("AZURE_SQL_SERVER")
        database = os.getenv("AZURE_SQL_DATABASE")
        username = os.getenv("AZURE_SQL_USERNAME")
        password = os.getenv("AZURE_SQL_PASSWORD")
        driver = os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")
        encrypt = os.getenv("AZURE_SQL_ENCRYPT", "yes")
        trust_cert = os.getenv("AZURE_SQL_TRUST_CERT", "no")

        if not all([server, database, username, password]):
            return None

        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"Encrypt={encrypt};"
            f"TrustServerCertificate={trust_cert};"
            "Connection Timeout=30;"
        )

    def _connect(self):
        if self.pyodbc is None:
            raise RuntimeError(self.last_error or "pyodbc is unavailable")
        if not self.connection_string:
            raise RuntimeError("Azure SQL environment variables are not configured")
        return self.pyodbc.connect(self.connection_string, autocommit=True)

    def init_schema(self) -> bool:
        """Create the prediction table if Azure SQL is reachable."""
        create_sql = """
        IF NOT EXISTS (
            SELECT 1 FROM sys.tables WHERE name = 'fraud_predictions'
        )
        BEGIN
            CREATE TABLE dbo.fraud_predictions (
                id INT IDENTITY(1,1) PRIMARY KEY,
                created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
                transaction_amount FLOAT NULL,
                product_code NVARCHAR(20) NULL,
                prediction INT NOT NULL,
                fraud_probability FLOAT NOT NULL,
                confidence FLOAT NOT NULL,
                risk_level NVARCHAR(20) NOT NULL,
                top_features NVARCHAR(MAX) NULL,
                request_json NVARCHAR(MAX) NOT NULL,
                response_json NVARCHAR(MAX) NOT NULL
            );
        END
        """

        try:
            with self._connect() as conn:
                conn.cursor().execute(create_sql)
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def status(self) -> Dict[str, Any]:
        ok = self.init_schema()
        return {
            "connected": ok,
            "provider": "Azure SQL" if ok else "memory fallback",
            "server": os.getenv("AZURE_SQL_SERVER"),
            "database": os.getenv("AZURE_SQL_DATABASE"),
            "last_error": None if ok else self.last_error,
            "memory_rows": len(self.memory_rows),
        }

    def save_prediction(self, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        row = {
            "id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "transaction_amount": request_data.get("TransactionAmt"),
            "product_code": request_data.get("ProductCD"),
            "prediction": response_data.get("prediction"),
            "fraud_probability": response_data.get("fraud_probability"),
            "confidence": response_data.get("confidence"),
            "risk_level": response_data.get("risk_level"),
            "top_features": response_data.get("top_features", []),
            "request": request_data,
            "response": response_data,
        }

        with self.lock:
            self.memory_rows.insert(0, row)
            del self.memory_rows[self.max_memory_rows :]

        insert_sql = """
        INSERT INTO dbo.fraud_predictions (
            transaction_amount, product_code, prediction, fraud_probability,
            confidence, risk_level, top_features, request_json, response_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self._connect() as conn:
                conn.cursor().execute(
                    insert_sql,
                    row["transaction_amount"],
                    row["product_code"],
                    row["prediction"],
                    row["fraud_probability"],
                    row["confidence"],
                    row["risk_level"],
                    json.dumps(row["top_features"]),
                    json.dumps(request_data),
                    json.dumps(response_data),
                )
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)

    def recent_predictions(self, limit: int = 50, only_alerts: bool = False) -> List[Dict[str, Any]]:
        limit = max(1, min(limit, 200))
        if self.connection_string and self.pyodbc is not None:
            try:
                where = "WHERE prediction = 1 OR risk_level IN ('High', 'Critical')" if only_alerts else ""
                query = f"""
                SELECT TOP ({limit})
                    id, created_at, transaction_amount, product_code, prediction,
                    fraud_probability, confidence, risk_level, top_features,
                    request_json, response_json
                FROM dbo.fraud_predictions
                {where}
                ORDER BY created_at DESC, id DESC
                """
                with self._connect() as conn:
                    rows = conn.cursor().execute(query).fetchall()

                result = []
                for row in rows:
                    result.append(
                        {
                            "id": row.id,
                            "created_at": row.created_at.isoformat(),
                            "transaction_amount": row.transaction_amount,
                            "product_code": row.product_code,
                            "prediction": row.prediction,
                            "fraud_probability": row.fraud_probability,
                            "confidence": row.confidence,
                            "risk_level": row.risk_level,
                            "top_features": json.loads(row.top_features or "[]"),
                            "request": json.loads(row.request_json),
                            "response": json.loads(row.response_json),
                        }
                    )
                self.last_error = None
                return result
            except Exception as exc:
                self.last_error = str(exc)

        with self.lock:
            rows = list(self.memory_rows)

        if only_alerts:
            rows = [
                row
                for row in rows
                if row["prediction"] == 1 or row["risk_level"] in {"High", "Critical"}
            ]
        return rows[:limit]
