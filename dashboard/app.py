"""
MLOps Monitoring Dashboard
Streamlit app with prediction input, drift visualization, and monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime
from pathlib import Path
import sqlite3
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "predictions.db"
PRODUCTION_MODEL_DIR = PROJECT_ROOT / "models" / "production"

# Add project root to path for drift imports
sys.path.insert(0, str(PROJECT_ROOT))


# ── Data helpers ──────────────────────────────────────────────

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
        df['input_data'] = df['input_data'].apply(json.loads)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for feature in ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                        'Population', 'AveOccup', 'Latitude', 'Longitude']:
            df[feature] = df['input_data'].apply(lambda x, f=feature: x.get(f, 0))

    return df


def get_model_metadata() -> dict:
    """Load production model metadata."""
    metadata_path = PRODUCTION_MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r") as f:
        return json.load(f)


# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="MLOps Dashboard",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 MLOps Monitoring Dashboard")
st.markdown("Real-time monitoring for California Housing price predictions")

# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    if auto_refresh:
        st.rerun()

    api_url = st.text_input("API URL", value="http://localhost:8000")

    st.markdown("---")
    st.header("📊 Model Info")
    metadata = get_model_metadata()
    if metadata:
        st.metric("Model Version",
                  f"v{metadata.get('source_version', metadata.get('version', 'N/A'))}")
        r2 = metadata.get('metrics', {}).get('r2', None)
        if r2 is not None:
            st.metric("R² Score", f"{r2:.4f}")
        st.text(f"Trained: {metadata.get('trained_at', 'N/A')[:10]}")
    else:
        st.warning("No production model found")

# ── Tabs ──────────────────────────────────────────────────────

tab_monitor, tab_predict, tab_drift = st.tabs([
    "📊 Monitor", "🔮 Predict", "🔍 Drift Analysis"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — MONITOR (existing dashboard)
# ═══════════════════════════════════════════════════════════════

with tab_monitor:
    df = get_predictions()

    if df.empty:
        st.info("👋 No predictions yet. Use the **Predict** tab or call the API!")
        st.code("curl -X POST http://localhost:8000/predict ...", language="bash")
    else:
        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", len(df))
        c2.metric("Avg Prediction", f"${df['prediction'].mean() * 100:.0f}K")
        c3.metric("Min", f"${df['prediction'].min() * 100:.0f}K")
        c4.metric("Max", f"${df['prediction'].max() * 100:.0f}K")

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

        # Feature analysis
        st.subheader("🔍 Feature Analysis")
        fcols = st.columns(4)
        for i, feat in enumerate(['MedInc', 'HouseAge', 'Population', 'AveRooms']):
            with fcols[i]:
                st.metric(f"Avg {feat}", f"{df[feat].mean():.2f}")

        st.markdown("---")

        # Predictions table
        st.subheader("📋 Recent Predictions")
        display_df = df[['id', 'prediction', 'model_version', 'timestamp',
                         'MedInc', 'HouseAge']].copy()
        display_df['prediction'] = display_df['prediction'].apply(lambda x: f"${x * 100:.0f}K")
        display_df.columns = ['ID', 'Prediction', 'Model', 'Timestamp', 'Income', 'Age']
        st.dataframe(display_df, use_container_width=True, height=400)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — PREDICT (user input → API → result)
# ═══════════════════════════════════════════════════════════════

with tab_predict:
    st.subheader("🔮 Make a Prediction")
    st.markdown("Enter housing features to get a predicted median house value.")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            med_inc = st.number_input("💰 Median Income (x$10K)",
                                      min_value=0.0, max_value=20.0, value=5.0, step=0.5,
                                      help="Median household income in block group (tens of thousands)")
            house_age = st.number_input("🏗️ House Age (years)",
                                        min_value=1.0, max_value=52.0, value=28.0, step=1.0,
                                        help="Median house age in block group")
            ave_rooms = st.number_input("🛋️ Average Rooms",
                                        min_value=1.0, max_value=15.0, value=5.5, step=0.5,
                                        help="Average rooms per household")
            ave_bedrms = st.number_input("🛏️ Average Bedrooms",
                                         min_value=0.5, max_value=5.0, value=1.0, step=0.1,
                                         help="Average bedrooms per household")

        with col_b:
            population = st.number_input("👥 Population",
                                         min_value=1.0, max_value=40000.0, value=1400.0, step=100.0,
                                         help="Block group population")
            ave_occup = st.number_input("👨‍👩‍👧‍👦 Avg Occupancy",
                                        min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                                        help="Average household members")
            latitude = st.number_input("📍 Latitude",
                                       min_value=32.0, max_value=42.0, value=34.0, step=0.1)
            longitude = st.number_input("📍 Longitude",
                                        min_value=-125.0, max_value=-114.0, value=-118.0, step=0.1)

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        payload = {
            "MedInc": med_inc, "HouseAge": house_age,
            "AveRooms": ave_rooms, "AveBedrms": ave_bedrms,
            "Population": population, "AveOccup": ave_occup,
            "Latitude": latitude, "Longitude": longitude,
        }

        with st.spinner("Calling prediction API..."):
            try:
                resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    price = result["prediction"] * 100_000

                    st.success(f"🏠 Predicted House Value: **${price:,.0f}**")

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("💵 Price", f"${price:,.0f}")
                    rc2.metric("🤖 Model", f"v{result['model_version']}")
                    rc3.metric("🆔 Prediction ID", result["prediction_id"])

                    with st.expander("📄 Input Summary"):
                        st.json(payload)
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the server is running:\n"
                         "`uvicorn api.main:app --reload`")
            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — DRIFT ANALYSIS
# ═══════════════════════════════════════════════════════════════

with tab_drift:
    st.subheader("🔍 Data Drift Analysis")
    st.markdown("Compare recent predictions against training data distribution.")

    try:
        from drift.drift_check import (
            load_reference_data, ks_drift_check, psi_drift_check,
            FEATURE_NAMES, KS_P_VALUE_THRESHOLD, PSI_THRESHOLD, DRIFT_FEATURE_RATIO,
        )

        pred_df = get_predictions()

        if pred_df.empty or len(pred_df) < 5:
            st.warning("⚠️ Need at least 5 predictions to run drift analysis. "
                       "Use the **Predict** tab to add more data.")
        else:
            if st.button("🔄 Run Drift Analysis", use_container_width=True):
                with st.spinner("Analyzing drift..."):
                    reference_df = load_reference_data()

                    # Build current data from predictions
                    current_df = pred_df[FEATURE_NAMES].copy()

                    ks_results = ks_drift_check(reference_df, current_df)
                    psi_results = psi_drift_check(reference_df, current_df)

                    ks_drifted = [f for f, r in ks_results.items() if r["drift_detected"]]
                    psi_drifted = [f for f, r in psi_results.items() if r["drift_detected"]]
                    total = len(FEATURE_NAMES)
                    ks_ratio = len(ks_drifted) / total
                    psi_ratio = len(psi_drifted) / total
                    drift_detected = (ks_ratio >= DRIFT_FEATURE_RATIO) or \
                                     (psi_ratio >= DRIFT_FEATURE_RATIO)

                # ── Status banner ──
                if drift_detected:
                    st.error("🚨 **DRIFT DETECTED** — Data distribution has shifted significantly!")
                else:
                    st.success("✅ **No Drift** — Data distribution is stable.")

                # ── Summary metrics ──
                s1, s2, s3 = st.columns(3)
                s1.metric("Predictions Analyzed", len(current_df))
                s2.metric("KS Drifted", f"{len(ks_drifted)}/{total}")
                s3.metric("PSI Drifted", f"{len(psi_drifted)}/{total}")

                st.markdown("---")

                # ── Overlaid distribution plots ──
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use("Agg")

                st.subheader("📊 Feature Distributions: Reference vs Current")
                st.markdown("🔵 **Reference (training)** &nbsp; 🔴 **Current (predictions)**")

                fig, axes = plt.subplots(2, 4, figsize=(16, 7))
                fig.patch.set_facecolor("#0e1117")
                axes = axes.flatten()

                for i, feature in enumerate(FEATURE_NAMES):
                    ax = axes[i]
                    ax.set_facecolor("#1a1c23")

                    ref_vals = reference_df[feature].dropna().values
                    cur_vals = current_df[feature].dropna().values

                    # Shared bins
                    all_vals = np.concatenate([ref_vals, cur_vals])
                    bins = np.linspace(all_vals.min(), all_vals.max(), 30)

                    ax.hist(ref_vals, bins=bins, alpha=0.55, color="#4A90D9",
                            label="Reference", density=True, edgecolor="none")
                    ax.hist(cur_vals, bins=bins, alpha=0.65, color="#E74C3C",
                            label="Current", density=True, edgecolor="none")

                    # Drift badge
                    drifted = ks_results[feature]["drift_detected"]
                    badge = "🚨 DRIFT" if drifted else "✅ OK"
                    ax.set_title(f"{feature}  {badge}", fontsize=11,
                                 fontweight="bold", color="white", pad=8)

                    ax.tick_params(colors="#888", labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_color("#333")

                axes[0].legend(fontsize=9, loc="upper right",
                               facecolor="#1a1c23", edgecolor="#444",
                               labelcolor="white")

                fig.tight_layout(pad=2.0)
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("---")

                # ── KS + PSI summary charts side by side ──
                col_ks, col_psi = st.columns(2)

                with col_ks:
                    st.subheader("� KS Statistic")
                    chart_df = pd.DataFrame({
                        "Feature": list(ks_results.keys()),
                        "KS Statistic": [r["statistic"] for r in ks_results.values()],
                    }).set_index("Feature")
                    st.bar_chart(chart_df)
                    st.caption(f"Threshold: p-value < {KS_P_VALUE_THRESHOLD}")

                with col_psi:
                    st.subheader("📈 PSI Score")
                    psi_df = pd.DataFrame({
                        "Feature": list(psi_results.keys()),
                        "PSI": [r["psi"] for r in psi_results.values()],
                    }).set_index("Feature")
                    st.bar_chart(psi_df)
                    st.caption(f"Threshold: PSI > {PSI_THRESHOLD}")

                st.markdown("---")

                # ── Detailed table ──
                st.subheader("📋 Detailed Results")
                detail_df = pd.DataFrame({
                    "Feature": FEATURE_NAMES,
                    "KS Stat": [ks_results[f]["statistic"] for f in FEATURE_NAMES],
                    "KS p-value": [ks_results[f]["p_value"] for f in FEATURE_NAMES],
                    "KS Drift": ["🚨" if ks_results[f]["drift_detected"] else "✅"
                                 for f in FEATURE_NAMES],
                    "PSI": [psi_results[f]["psi"] for f in FEATURE_NAMES],
                    "PSI Drift": ["🚨" if psi_results[f]["drift_detected"] else "✅"
                                  for f in FEATURE_NAMES],
                })
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

    except ImportError as e:
        st.error(f"Cannot load drift module: {e}")
    except FileNotFoundError:
        st.warning("⚠️ Reference data not found. Run training first: "
                   "`python -m training.train`")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
