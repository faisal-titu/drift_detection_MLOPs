#!/bin/bash
set -e

echo "Starting MLOps Drift Detection System..."

# Create data directory if it doesn't exist
mkdir -p /app/data /app/logs

# Start FastAPI in the background
echo "Starting FastAPI server on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
echo "Waiting for API to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit on port 7860 (HF Spaces requirement)
echo "Starting Streamlit dashboard on port 7860..."
exec streamlit run dashboard/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none
