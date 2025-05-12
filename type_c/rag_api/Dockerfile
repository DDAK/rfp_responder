FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential

# Install spaCy first
RUN pip install --no-cache-dir spacy

# Download the language model correctly
RUN python -m spacy download en_core_web_sm

# Install remaining requirements
RUN pip install --no-cache-dir --retries 5 --timeout 100 -r requirements.txt

# Clean up build dependencies
RUN apt-get purge -y --auto-remove gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY main.py ./
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]