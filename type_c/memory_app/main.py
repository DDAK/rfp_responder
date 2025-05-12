from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, requests, redis, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("memory-app")

# ── Config ─────────────────────────────────────────────────────────────
REDIS = redis.Redis(host="redis", port=6379, decode_responses=True)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL     = os.getenv("OLLAMA_MODEL", "llama3:8b")

MAX_TURNS = 6
SUMMARY_AFTER = 4

# ── Helpers ────────────────────────────────────────────────────────────
def redis_key(sess, suffix): return f"chat:{sess}:{suffix}"

def ollama_chat(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}]
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    data = r.json()
    if "message" not in data:           # new schema? error blob?
        raise ValueError(f"Ollama error: {data}")
    return data["message"]["content"]

def summarise(txt: str) -> str:
    return ollama_chat(f"Summarise in one paragraph:\n\n{txt}")

# ── FastAPI ────────────────────────────────────────────────────────────
app = FastAPI(title="In‑Memory Chat")

class ChatReq(BaseModel):
    user: str

@app.post("/chat/{session_id}")
def chat(session_id: str, req: ChatReq):
    try:
        turns_key = redis_key(session_id, "turns")
        sum_key   = redis_key(session_id, "summary")

        turns = REDIS.lrange(turns_key, 0, -1)
        context = (REDIS.get(sum_key) or "") + "\n".join(turns)
        prompt = f"{context}\nUser: {req.user}\nAssistant:"

        reply = ollama_chat(prompt)

        # store turn
        REDIS.rpush(turns_key, f"User:{req.user}", f"Assistant:{reply}")
        REDIS.ltrim(turns_key, -MAX_TURNS*2, -1)

        # summarise if needed
        if REDIS.llen(turns_key) // 2 >= SUMMARY_AFTER:
            history = "\n".join(REDIS.lrange(turns_key, 0, -1))
            REDIS.set(sum_key, summarise(history))

        return {"reply": reply}

    except Exception as e:
        log.exception("chat handler error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
def history(session_id: str):
    return {
        "summary": REDIS.get(redis_key(session_id, "summary")),
        "turns":   REDIS.lrange(redis_key(session_id, "turns"), 0, -1)
    }
