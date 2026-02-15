FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY api/ ./api/
COPY training/ ./training/
COPY drift/ ./drift/
COPY registry/ ./registry/
COPY dashboard/ ./dashboard/
COPY utils/ ./utils/
COPY conftest.py .
COPY pyproject.toml .

# Copy trained model and data
COPY models/production/ ./models/production/
COPY data/reference_data.csv ./data/reference_data.csv

# Create writable directories
RUN mkdir -p data logs

# Make startup script executable
COPY start.sh .
RUN chmod +x start.sh

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["./start.sh"]
