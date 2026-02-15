"""
Database Module - SQLite for Prediction Logging
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "predictions.db"


def get_connection() -> sqlite3.Connection:
    """Get SQLite database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_data TEXT NOT NULL,
            prediction REAL NOT NULL,
            model_version INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized: %s", DB_PATH)


def log_prediction(
    input_data: Dict,
    prediction: float,
    model_version: int
) -> int:
    """
    Log a prediction to the database.
    
    Args:
        input_data: Input features as dictionary
        prediction: Model prediction
        model_version: Version of model used
    
    Returns:
        Prediction ID
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO predictions (input_data, prediction, model_version, timestamp)
        VALUES (?, ?, ?, ?)
    """, (json.dumps(input_data), prediction, model_version, datetime.now().isoformat()))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id


def get_predictions(limit: int = 100) -> List[Dict]:
    """
    Get recent predictions.
    
    Args:
        limit: Maximum number of predictions to return
    
    Returns:
        List of prediction records
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, input_data, prediction, model_version, timestamp
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row["id"],
            "input_data": json.loads(row["input_data"]),
            "prediction": row["prediction"],
            "model_version": row["model_version"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def get_prediction_count() -> int:
    """Get total number of predictions."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    
    return count


# Initialize on import
init_db()
