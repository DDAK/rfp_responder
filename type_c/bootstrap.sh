#!/usr/bin/env bash
set -e

echo "🔧 Initialising local volumes & downloading the Llama‑3 8B model…"

# 1. create seed dirs (if they don't exist)
mkdir -p seed/airflow/{dags,logs,plugins}
mkdir -p seed/ollama

# 2. pull images
docker compose pull

# 3. start only Ollama
docker compose up -d ollama
echo "⌛ Waiting for Ollama to start…"; sleep 5

# 4. find the actual container ID for the 'ollama' service
OLLAMA_CID=$(docker compose ps -q ollama)

# 5. download model once
if ! docker compose exec ollama ollama list | grep -q "llama3:8b"; then
  docker compose exec ollama ollama pull llama3:8b  # gte-qwen2-1.5b-instruct-embed-f16
fi

echo "✅ Bootstrap finished. Run 'docker compose up -d' for the full stack."
