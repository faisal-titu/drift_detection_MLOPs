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
import logging
import warnings
import queue
import threading
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "predictions.db"
PRODUCTION_MODEL_DIR = PROJECT_ROOT / "models" / "production"

# Add project root to path for drift imports
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# -- Data helpers -----------------------------------------------------------

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


def get_api_health(api_url: str) -> dict:
    """Fetch API health safely."""
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"status": "unreachable", "model_loaded": False, "model_version": None}


def _init_simulation_state() -> None:
    """Initialize persistent session state for streaming simulation."""
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "sim_queue" not in st.session_state:
        st.session_state.sim_queue = queue.Queue()
    if "sim_events" not in st.session_state:
        st.session_state.sim_events = []
    if "sim_summary" not in st.session_state:
        st.session_state.sim_summary = None
    if "sim_last_batch" not in st.session_state:
        st.session_state.sim_last_batch = None
    if "sim_thread" not in st.session_state:
        st.session_state.sim_thread = None
    if "sim_stop_event" not in st.session_state:
        st.session_state.sim_stop_event = None
    if "sim_config" not in st.session_state:
        st.session_state.sim_config = None


def _run_stream_worker(config: dict, event_queue: "queue.Queue", stop_event: threading.Event) -> None:
    """Background worker to run synthetic streaming and emit status events."""
    try:
        from drift.stream_synthetic import stream_and_monitor

        def progress_callback(event: dict) -> None:
            event_queue.put(event)

        stream_and_monitor(
            num_batches=config["num_batches"],
            batch_size=config["batch_size"],
            drift_start_batch=config["drift_start_batch"],
            performance_r2_threshold=config["performance_r2_threshold"],
            min_r2_improvement=config["min_r2_improvement"],
            sleep_seconds=config["sleep_seconds"],
            retrain_cooldown_batches=config["retrain_cooldown_batches"],
            random_state=config["random_state"],
            max_retrains=config.get("max_retrains", 3),
            ramp_batches=config.get("ramp_batches", 8),
            progress_callback=progress_callback,
            stop_event=stop_event,
        )
    except Exception as exc:
        event_queue.put({"event": "error", "message": str(exc)})
    finally:
        event_queue.put({"event": "worker_done"})


def _drain_simulation_events() -> None:
    """Drain queued worker events into session state."""
    _init_simulation_state()
    event_queue = st.session_state.sim_queue

    while not event_queue.empty():
        event = event_queue.get_nowait()
        event["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if event.get("event") == "batch":
            st.session_state.sim_last_batch = event
            st.session_state.sim_events.append(event)
        elif event.get("event") in {"retrain", "retrain_skipped", "error", "stopped"}:
            st.session_state.sim_events.append(event)
        elif event.get("event") == "summary":
            st.session_state.sim_summary = event
            st.session_state.sim_events.append(event)
            st.session_state.sim_running = False
        elif event.get("event") == "worker_done":
            st.session_state.sim_running = False

    if len(st.session_state.sim_events) > 500:
        st.session_state.sim_events = st.session_state.sim_events[-500:]


# -- Page config ------------------------------------------------------------

st.set_page_config(
    page_title="MLOps Control Center",
    page_icon="📈",
    layout="wide",
)

_init_simulation_state()
_drain_simulation_events()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
    .status-ok {padding: 0.35rem 0.6rem; border-radius: 0.5rem; background: #163d2a; color: #7ee2b8; font-weight: 600;}
    .status-warn {padding: 0.35rem 0.6rem; border-radius: 0.5rem; background: #4a3318; color: #ffc166; font-weight: 600;}
    .status-err {padding: 0.35rem 0.6rem; border-radius: 0.5rem; background: #4c1f1f; color: #ff9b9b; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MLOps Control Center")
st.markdown("Production monitoring, drift operations, and retraining controls")

# -- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    refresh_now = st.button("Refresh Now", use_container_width=True)
    if refresh_now:
        st.rerun()

    api_url = st.text_input("API URL", value="http://localhost:8000")
    api_health = get_api_health(api_url)

    st.markdown("---")
    st.subheader("API Health")
    if api_health.get("status") == "healthy":
        st.markdown('<div class="status-ok">Healthy</div>', unsafe_allow_html=True)
    elif api_health.get("status") == "degraded":
        st.markdown('<div class="status-warn">Degraded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">Unreachable</div>', unsafe_allow_html=True)

    st.caption(f"Model loaded: {api_health.get('model_loaded', False)}")
    st.caption(f"Serving version: {api_health.get('model_version', 'N/A')}")

    st.markdown("---")
    st.header("Model Info")
    metadata = get_model_metadata()
    if metadata:
        st.metric("Model Version",
                  f"v{metadata.get('source_version', metadata.get('version', 'N/A'))}")
        r2 = metadata.get('metrics', {}).get('r2', None)
        if r2 is not None:
            st.metric("R2 Score", f"{r2:.4f}")
        st.text(f"Trained: {metadata.get('trained_at', 'N/A')[:10]}")
    else:
        st.warning("No production model found")

# -- Tabs -------------------------------------------------------------------

tab_monitor, tab_predict, tab_drift, tab_ops = st.tabs([
    "Monitor", "Predict", "Drift Analysis", "Operations"
])

# ===================================================================
# TAB 1  MONITOR
# ===================================================================

with tab_monitor:
    df = get_predictions()

    if df.empty:
        st.info("No predictions yet. Use the **Predict** tab or call the API!")
        st.code("curl -X POST http://localhost:8000/predict ...", language="bash")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", len(df))
        c2.metric("Avg Prediction", f"${df['prediction'].mean() * 100:.0f}K")
        c3.metric("Min", f"${df['prediction'].min() * 100:.0f}K")
        c4.metric("Max", f"${df['prediction'].max() * 100:.0f}K")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Distribution")
            st.bar_chart(df['prediction'].value_counts(bins=20).sort_index())
        with col2:
            st.subheader("Predictions Over Time")
            if len(df) > 1:
                time_data = df.set_index('timestamp')['prediction'].resample('1h').mean()
                st.line_chart(time_data)
            else:
                st.info("Need more predictions to show time series")

        st.markdown("---")

        st.subheader("Feature Analysis")
        fcols = st.columns(4)
        for i, feat in enumerate(['MedInc', 'HouseAge', 'Population', 'AveRooms']):
            with fcols[i]:
                st.metric(f"Avg {feat}", f"{df[feat].mean():.2f}")

        st.markdown("---")

        st.subheader("Recent Predictions")
        display_df = df[['id', 'prediction', 'model_version', 'timestamp',
                         'MedInc', 'HouseAge']].copy()
        display_df['prediction'] = display_df['prediction'].apply(lambda x: f"${x * 100:.0f}K")
        display_df.columns = ['ID', 'Prediction', 'Model', 'Timestamp', 'Income', 'Age']
        st.dataframe(display_df, use_container_width=True, height=400)

# ===================================================================
# TAB 2  PREDICT
# ===================================================================

with tab_predict:
    st.subheader("Make a Prediction")
    st.markdown("Enter housing features to get a predicted median house value.")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            med_inc = st.slider("Median Income (x$10K)",
                                min_value=0.5, max_value=15.0, value=5.0, step=0.1,
                                help="Median household income in block group (tens of thousands)")
            house_age = st.slider("House Age (years)",
                                  min_value=1.0, max_value=52.0, value=28.0, step=1.0,
                                  help="Median house age in block group")
            ave_rooms = st.slider("Average Rooms",
                                  min_value=1.0, max_value=15.0, value=5.5, step=0.1,
                                  help="Average rooms per household")
            ave_bedrms = st.slider("Average Bedrooms",
                                   min_value=0.5, max_value=5.0, value=1.0, step=0.1,
                                   help="Average bedrooms per household")

        with col_b:
            population = st.slider("Population",
                                   min_value=1, max_value=40000, value=1400, step=50,
                                   help="Block group population")
            ave_occup = st.slider("Avg Occupancy",
                                  min_value=1.0, max_value=10.0, value=3.0, step=0.1,
                                  help="Average household members")
            latitude = st.slider("Latitude",
                                 min_value=32.0, max_value=42.0, value=34.0, step=0.1)
            longitude = st.slider("Longitude",
                                  min_value=-125.0, max_value=-114.0, value=-118.0, step=0.1)

        submitted = st.form_submit_button("Predict", use_container_width=True)

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

                    st.success(f"Predicted House Value: **${price:,.0f}**")

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Price", f"${price:,.0f}")
                    rc2.metric("Model", f"v{result['model_version']}")
                    rc3.metric("Prediction ID", result["prediction_id"])

                    with st.expander("Input Summary"):
                        st.json(payload)
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.ConnectionError:
                st.error("Cannot connect to API. Make sure the server is running:\n"
                         "`uvicorn api.main:app --reload`")
            except Exception as e:
                st.error(f"Error: {e}")

# ===================================================================
# TAB 3  DRIFT ANALYSIS
# ===================================================================

with tab_drift:
    st.subheader("Data Drift Analysis")
    st.markdown("Compare recent predictions against training data distribution.")

    try:
        from drift.drift_check import (
            load_reference_data, ks_drift_check, psi_drift_check,
            FEATURE_NAMES, KS_P_VALUE_THRESHOLD, PSI_THRESHOLD, DRIFT_FEATURE_RATIO,
        )

        pred_df = get_predictions()

        if pred_df.empty or len(pred_df) < 5:
            st.warning("Need at least 5 predictions to run drift analysis. "
                       "Use the **Predict** tab to add more data.")
        else:
            if st.button("Run Drift Analysis", use_container_width=True):
                with st.spinner("Analyzing drift..."):
                    reference_df = load_reference_data()

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

                # -- Status banner --
                if drift_detected:
                    st.error("**DRIFT DETECTED** - Data distribution has shifted significantly!")
                else:
                    st.success("**No Drift** - Data distribution is stable.")

                # -- Summary metrics --
                s1, s2, s3 = st.columns(3)
                s1.metric("Predictions Analyzed", len(current_df))
                s2.metric("KS Drifted", f"{len(ks_drifted)}/{total}")
                s3.metric("PSI Drifted", f"{len(psi_drifted)}/{total}")

                st.markdown("---")

                # -- Overlaid distribution plots --
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use("Agg")

                st.subheader("Feature Distributions: Reference vs Current")
                st.markdown("Blue = **Reference (training)** | Red = **Current (predictions)**")

                fig, axes = plt.subplots(2, 4, figsize=(16, 7))
                fig.patch.set_facecolor("#0e1117")
                axes = axes.flatten()

                for i, feature in enumerate(FEATURE_NAMES):
                    ax = axes[i]
                    ax.set_facecolor("#1a1c23")

                    ref_vals = reference_df[feature].dropna().values
                    cur_vals = current_df[feature].dropna().values

                    all_vals = np.concatenate([ref_vals, cur_vals])
                    bins = np.linspace(all_vals.min(), all_vals.max(), 30)

                    ax.hist(ref_vals, bins=bins, alpha=0.55, color="#4A90D9",
                            label="Reference", density=True, edgecolor="none")
                    ax.hist(cur_vals, bins=bins, alpha=0.65, color="#E74C3C",
                            label="Current", density=True, edgecolor="none")

                    drifted = ks_results[feature]["drift_detected"]
                    badge = "DRIFT" if drifted else "OK"
                    ax.set_title(f"{feature}  [{badge}]", fontsize=11,
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

                # -- KS + PSI summary charts side by side --
                col_ks, col_psi = st.columns(2)

                with col_ks:
                    st.subheader("KS Statistic")
                    chart_df = pd.DataFrame({
                        "Feature": list(ks_results.keys()),
                        "KS Statistic": [r["statistic"] for r in ks_results.values()],
                    }).set_index("Feature")
                    st.bar_chart(chart_df)
                    st.caption(f"Threshold: p-value < {KS_P_VALUE_THRESHOLD}")

                with col_psi:
                    st.subheader("PSI Score")
                    psi_df = pd.DataFrame({
                        "Feature": list(psi_results.keys()),
                        "PSI": [r["psi"] for r in psi_results.values()],
                    }).set_index("Feature")
                    st.bar_chart(psi_df)
                    st.caption(f"Threshold: PSI > {PSI_THRESHOLD}")

                st.markdown("---")

                # -- Detailed table --
                st.subheader("Detailed Results")
                detail_df = pd.DataFrame({
                    "Feature": FEATURE_NAMES,
                    "KS Stat": [ks_results[f]["statistic"] for f in FEATURE_NAMES],
                    "KS p-value": [ks_results[f]["p_value"] for f in FEATURE_NAMES],
                    "KS Drift": ["Yes" if ks_results[f]["drift_detected"] else "No"
                                 for f in FEATURE_NAMES],
                    "PSI": [psi_results[f]["psi"] for f in FEATURE_NAMES],
                    "PSI Drift": ["Yes" if psi_results[f]["drift_detected"] else "No"
                                  for f in FEATURE_NAMES],
                })
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

    except ImportError as e:
        st.error(f"Cannot load drift module: {e}")
    except FileNotFoundError:
        st.warning("Reference data not found. Run training first: "
                   "`python -m training.train`")

# ===================================================================
# TAB 4  OPERATIONS (LIVE STREAMING + RETRAIN STATUS)
# ===================================================================

with tab_ops:
    st.subheader("Streaming Simulation & Retrain Operations")
    st.markdown(
        "Run synthetic concept-drift streams and watch live trigger/retrain outcomes.  "
        "Drift ramps **gradually** to avoid overwhelming CPU/RAM."
    )

    # ---- live status banner ----
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)

        if st.session_state.sim_running:
            c1.markdown('<div class="status-warn">Running</div>', unsafe_allow_html=True)
        elif st.session_state.sim_summary:
            c1.markdown('<div class="status-ok">Finished</div>', unsafe_allow_html=True)
        else:
            c1.markdown('<div class="status-err">Idle</div>', unsafe_allow_html=True)

        latest_alerts = 0
        latest_retrains = 0
        latest_intensity = 0.0
        if st.session_state.sim_summary:
            latest_alerts = st.session_state.sim_summary.get("alerts", 0)
            latest_retrains = st.session_state.sim_summary.get("retrains", 0)
        elif st.session_state.sim_last_batch:
            latest_alerts = st.session_state.sim_last_batch.get("alerts", 0)
            latest_retrains = st.session_state.sim_last_batch.get("retrains", 0)
            latest_intensity = st.session_state.sim_last_batch.get("intensity", 0.0)

        c2.metric("Drift Alerts", latest_alerts)
        c3.metric("Retrains", latest_retrains)
        c4.metric("Drift Intensity", f"{latest_intensity:.0%}")

    # ---- configuration form ----
    with st.expander("Simulation Parameters", expanded=not st.session_state.sim_running):
        with st.form("stream_simulation_form"):
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown("**Stream Settings**")
                num_batches = st.number_input(
                    "Batches", min_value=5, max_value=200, value=20, step=5,
                    help="Total number of data batches to stream",
                )
                batch_size = st.number_input(
                    "Batch Size", min_value=50, max_value=2000, value=300, step=50,
                    help="Samples per batch (keep low to save RAM)",
                )
                drift_start_batch = st.number_input(
                    "Drift Start Batch", min_value=1, max_value=199, value=8, step=1,
                    help="Batch index where drift begins ramping",
                )
                ramp_batches = st.number_input(
                    "Ramp Batches", min_value=1, max_value=50, value=8, step=1,
                    help="Number of batches over which drift ramps from 0% to 100%",
                )
            with s2:
                st.markdown("**Retrain Guards**")
                performance_r2_threshold = st.slider(
                    "Perf R2 Threshold", min_value=-1.0, max_value=1.0, value=0.40, step=0.05,
                    help="Alert when production R2 drops below this",
                )
                min_r2_improvement = st.slider(
                    "Min R2 Improvement", min_value=0.0, max_value=0.2, value=0.02, step=0.005,
                    help="New model must beat current R2 by at least this margin to deploy",
                )
                retrain_cooldown_batches = st.number_input(
                    "Retrain Cooldown (batches)", min_value=1, max_value=50, value=5, step=1,
                    help="Minimum batch gap between retrain attempts",
                )
                max_retrains = st.number_input(
                    "Max Retrains", min_value=1, max_value=10, value=3, step=1,
                    help="Hard cap on retrain attempts per simulation run",
                )
            with s3:
                st.markdown("**Pacing**")
                sleep_seconds = st.number_input(
                    "Sleep Between Batches (s)", min_value=0.5, max_value=10.0, value=1.0, step=0.5,
                    help="Pause between batches (min 0.5s enforced)",
                )
                random_state = st.number_input(
                    "Random Seed", min_value=1, max_value=999999, value=42, step=1,
                )

            start_clicked = st.form_submit_button(
                "Start Simulation", use_container_width=True,
                disabled=st.session_state.sim_running,
            )

    # ---- start / stop logic ----
    stop_clicked = st.button(
        "Stop Simulation", use_container_width=True,
        disabled=not st.session_state.sim_running,
    )

    if start_clicked and not st.session_state.sim_running:
        if drift_start_batch >= num_batches:
            st.error("Drift start batch must be less than total number of batches.")
        else:
            st.session_state.sim_events = []
            st.session_state.sim_summary = None
            st.session_state.sim_last_batch = None

            st.session_state.sim_config = {
                "num_batches": int(num_batches),
                "batch_size": int(batch_size),
                "drift_start_batch": int(drift_start_batch),
                "performance_r2_threshold": float(performance_r2_threshold),
                "min_r2_improvement": float(min_r2_improvement),
                "sleep_seconds": float(sleep_seconds),
                "retrain_cooldown_batches": int(retrain_cooldown_batches),
                "random_state": int(random_state),
                "max_retrains": int(max_retrains),
                "ramp_batches": int(ramp_batches),
            }

            stop_event = threading.Event()
            st.session_state.sim_stop_event = stop_event
            st.session_state.sim_running = True

            worker = threading.Thread(
                target=_run_stream_worker,
                args=(st.session_state.sim_config, st.session_state.sim_queue, stop_event),
                daemon=True,
            )
            st.session_state.sim_thread = worker
            worker.start()
            st.success("Streaming simulation started.")

    if stop_clicked and st.session_state.sim_stop_event is not None:
        st.session_state.sim_stop_event.set()
        st.warning("Stop requested — worker will finish current batch then halt.")

    # ---- live progress ----
    if st.session_state.sim_running and st.session_state.sim_config:
        batch_info = st.session_state.sim_last_batch
        if batch_info:
            total = st.session_state.sim_config["num_batches"]
            current = batch_info.get("batch_index", 0) + 1
            progress = current / total
            st.progress(
                min(max(progress, 0.0), 1.0),
                text=f"Batch {current}/{total}  |  "
                     f"drift intensity {batch_info.get('intensity', 0):.0%}  |  "
                     f"R2 {batch_info.get('r2', 0.0):.4f}  |  "
                     f"RMSE {batch_info.get('rmse', 0.0):.4f}",
            )
        else:
            st.info("Simulation is running — waiting for first batch...")

    # ---- event log ----
    st.markdown("---")
    st.subheader("Live Event Log")
    if st.session_state.sim_events:
        events_df = pd.DataFrame(st.session_state.sim_events)
        visible_cols = [
            c for c in [
                "timestamp", "event", "batch_index", "intensity", "phase",
                "r2", "rmse", "drift_detected", "perf_alert",
                "success", "reason", "max_retrains",
            ]
            if c in events_df.columns
        ]
        st.dataframe(events_df[visible_cols].tail(100), use_container_width=True, height=320)
    else:
        st.info("No operation events yet.")

    if st.session_state.sim_summary:
        summary = st.session_state.sim_summary
        st.success(
            f"**Simulation complete** — "
            f"alerts: {summary.get('alerts', 0)}  |  "
            f"retrain successes: {summary.get('retrains', 0)}"
        )

    # auto-refresh while running (non-blocking via st.rerun after short sleep)
    if st.session_state.sim_running:
        time.sleep(2)
        st.rerun()

# -- Footer -----------------------------------------------------------------
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
