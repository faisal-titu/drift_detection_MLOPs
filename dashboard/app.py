"""
MLOps Monitoring Dashboard
Streamlit app for visualizing predictions and system health.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import sqlite3

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "predictions.db"
PRODUCTION_MODEL_DIR = PROJECT_ROOT / "models" / "production"


def get_db_connection():
    """Get SQLite connection."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_predictions(limit: int = 500) -> pd.DataFrame:
    """Load predictions from database."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    query = """
        SELECT id, input_data, prediction, model_version, timestamp
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    
    if not df.empty:
        # Parse input_data JSON
        df['input_data'] = df['input_data'].apply(json.loads)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract features
        for feature in ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'Latitude', 'Longitude']:
            df[feature] = df['input_data'].apply(lambda x: x.get(feature, 0))
    
    return df


def get_model_metadata() -> dict:
    """Load production model metadata."""
    metadata_path = PRODUCTION_MODEL_DIR / "metadata.json"
    
    if not metadata_path.exists():
        return {}
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def main():
    # Page config
    st.set_page_config(
        page_title="MLOps Dashboard",
        page_icon="🏠",
        layout="wide",
    )
    
    # Header
    st.title("🏠 MLOps Monitoring Dashboard")
    st.markdown("Real-time monitoring for California Housing price predictions")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            st.rerun()
        
        st.markdown("---")
        st.header("📊 Model Info")
        metadata = get_model_metadata()
        if metadata:
            st.metric("Model Version", f"v{metadata.get('source_version', metadata.get('version', 'N/A'))}")
            st.metric("R² Score", f"{metadata.get('metrics', {}).get('r2', 'N/A'):.4f}")
            st.text(f"Trained: {metadata.get('trained_at', 'N/A')[:10]}")
        else:
            st.warning("No production model found")
    
    # Load data
    df = get_predictions()
    
    if df.empty:
        st.info("👋 No predictions yet. Start the API and make some predictions!")
        st.code("uvicorn api.main:app --reload", language="bash")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(df))
    
    with col2:
        st.metric("Avg Prediction", f"${df['prediction'].mean() * 100:.0f}K")
    
    with col3:
        st.metric("Min", f"${df['prediction'].min() * 100:.0f}K")
    
    with col4:
        st.metric("Max", f"${df['prediction'].max() * 100:.0f}K")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Prediction Distribution")
        st.bar_chart(df['prediction'].value_counts(bins=20).sort_index())
    
    with col2:
        st.subheader("📊 Predictions Over Time")
        if len(df) > 1:
            time_data = df.set_index('timestamp')['prediction'].resample('1H').mean()
            st.line_chart(time_data)
        else:
            st.info("Need more predictions to show time series")
    
    st.markdown("---")
    
    # Feature distribution
    st.subheader("🔍 Feature Analysis")
    feature_cols = st.columns(4)
    features = ['MedInc', 'HouseAge', 'Population', 'AveRooms']
    
    for i, feature in enumerate(features):
        with feature_cols[i]:
            st.metric(f"Avg {feature}", f"{df[feature].mean():.2f}")
    
    st.markdown("---")
    
    # Predictions table
    st.subheader("📋 Recent Predictions")
    
    # Display table
    display_df = df[['id', 'prediction', 'model_version', 'timestamp', 'MedInc', 'HouseAge']].copy()
    display_df['prediction'] = display_df['prediction'].apply(lambda x: f"${x * 100:.0f}K")
    display_df.columns = ['ID', 'Prediction', 'Model', 'Timestamp', 'Income', 'Age']
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
