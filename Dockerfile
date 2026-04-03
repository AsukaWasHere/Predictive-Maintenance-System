FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (important for ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip (important)
RUN pip install --upgrade pip

# Install dependencies with higher timeout (FIXES YOUR ERROR)
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY outputs/models/ outputs/models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]