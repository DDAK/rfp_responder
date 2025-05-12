# rag_service.py  – hybrid KG‑filtered Retrieval‑Augmented Generation
import os, json, re, asyncio, textwrap, math
import traceback
from typing import List
import psycopg2, spacy, tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
import groq
import redis, uuid

REDIS = redis.Redis(host="redis", port=6379, decode_responses=True)

# ── config --------------------------------------------------------
PG = dict(host="postgres", dbname="demo", user="demo", password="demo1234")
NEO = GraphDatabase.driver("bolt://neo4j:7687",
                           auth=("neo4j", os.getenv("NEO4J_PASSWORD","demo1234")))
EMBED_DIM  = 1536
EMBED_MODEL = "llama3:8b"           # Ollama embeddings (1536 after truncate)
CHAT_MODEL  = "llama-3.3-70b-versatile"  # Groq chat completion
GROQ = groq.Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

NLP  = spacy.load("en_core_web_sm")
TOK  = tiktoken.get_encoding("cl100k_base")

# ── request / response models ------------------------------------
def mem_key(session_id: str):
    return f"chat:{session_id}"

def save_turn(session_id: str, role: str, text: str):
    REDIS.rpush(mem_key(session_id), json.dumps({"role": role, "text": text}))

def load_history(session_id: str, max_turns: int = 20):
    items = REDIS.lrange(mem_key(session_id), -max_turns*2, -1)
    return [json.loads(i) for i in items]

# ── extend AskReq model -----------------------------------------
class AskReq(BaseModel):
    question: str
    session_id: str | None = None    # <- new
    k: int = 6

class AskResp(BaseModel):
    answer: str
    sources: List[str]

app = FastAPI()


# ── embedding helper (local Ollama) ------------------------------
import requests
def embed(text: str) -> List[float]:
    r = requests.post("http://ollama:11434/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text[:2000]},
                      timeout=60).json()
    if "embedding" not in r:
        raise RuntimeError(r)
    return r["embedding"][:EMBED_DIM]


# ── 1. Entity extractor ------------------------------------------
def extract_entities(q: str):
    doc = NLP(q)
    people = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
    nums   = re.findall(r"[A-Z]{2,5}-\d{2,}", q)      # crude case‑ID pattern
    orgs   = {ent.text for ent in doc.ents if ent.label_ in ("ORG","NORP")}
    return people, nums, orgs


# ── 2. Cypher filter ---------------------------------------------
def entities_to_case_ids(people, case_ids, orgs):
    ids = set(case_ids)
    with NEO.session() as ses:
        if people:
            rows = ses.run("""
              MATCH (p:Person)-[:HAS_GROSS_INCOME|HAS_NET_INCOME|EMPLOYED_BY]-()
              WHERE p.name IN $names
              RETURN DISTINCT p.case_id AS cid
            """, names=list(people))
            ids.update(r["cid"] for r in rows if r["cid"])
        if orgs:
            rows = ses.run("""
              MATCH (o:Org)<-[:EMPLOYED_BY]-(:Person)
              WHERE o.name IN $orgs
              RETURN DISTINCT o.case_id AS cid
            """, orgs=list(orgs))
            ids.update(r["cid"] for r in rows if r["cid"])
    return list(ids)


# ── 3. Vector search ---------------------------------------------
def similar_chunks(query_vec, k, scope_case_ids=None):
    limit = k * 4  # over-fetch
    if scope_case_ids:
        sql = """
          SELECT chunk_id, text, extra, (embedding <=> (%s::vector)) AS dist
          FROM   documents
          WHERE  case_id = ANY(%s)
          ORDER  BY dist
          LIMIT  %s;
        """
        params = [query_vec, scope_case_ids, limit]
    else:
        sql = """
          SELECT chunk_id, text, extra, (embedding <=> (%s::vector)) AS dist
          FROM   documents
          ORDER  BY dist
          LIMIT  %s;
        """
        params = [query_vec, limit]

    with psycopg2.connect(**PG) as pg, pg.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ── 4. Token budget & summarisation -------------------------------
MAX_TOK = 1500
def tokens(s: str) -> int:
    return len(TOK.encode(s))

def summarise(snippet: str) -> str:
    prompt = ("Summarise the following legal snippet in <=30 words, "
              "keep numbers exact:\n```" + snippet[:1200] + "```")
    chat = GROQ.chat.completions.create(model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}], stream=False)
    return chat.choices[0].message.content.strip()

def build_context(snips):
    # snips is a list of tuples: (chunk_id, text, extra, dist)
    budget = MAX_TOK - 350      # leave space for system + user tokens
    selected = []
    total = 0

    for sid, txt, extra, _ in snips:
        seg  = txt.strip().replace("\n", " ")
        cost = tokens(seg)

        # if this snippet would bust the budget, summarise it
        if total + cost > budget:
            seg  = summarise(seg)
            cost = tokens(seg)

        # if it still doesn’t fit, skip it
        if total + cost > budget:
            continue

        selected.append((sid, seg))
        total += cost

        # stop once we hit the budget
        if total >= budget:
            break

    return selected


# ── 5. Prompt composer -------------------------------------------
SYSTEM = """
You are an AI assistant that accelerates the social security department's application processing 
from 5-20 days to under 30 minutes with 90% automation. You process multilingual documents (English/Arabic), 
extract relevant information, and generate personalized support recommendations based on retrieved applicant 
data, financial records, eligibility criteria, and available programs. You can only use information from 
your context window or through your retrieval system. Dont mention you are using context to give details in your output.
"""



def compose_prompt(question, context_chunks):
    snippets_txt = "\n\n".join(
        f"[{sid}] {textwrap.shorten(txt, 300)}"
        for sid, txt in context_chunks)
    return [
        {"role":"system","content":SYSTEM},
        {"role":"assistant","content":"Context snippets:\n"+snippets_txt},
        {"role":"user","content":question}
    ]


# ── 6. Endpoint ---------------------------------------------------
@app.post("/ask", response_model=AskResp)
async def ask(req: AskReq):
    try:
        session_id = req.session_id or str(uuid.uuid4())

        # 0. add previous chat to the prompt (optional, 4‑turn window)
        history = load_history(session_id, max_turns=4)
        history_msgs = [{"role": h["role"], "content": h["text"]} for h in history]

        q_vec = embed(req.question)
        people, cids, orgs = extract_entities(req.question)
        scope_ids = entities_to_case_ids(people, cids, orgs)

        print("Question:", req.question)
        print("people: %s, cids: %s, orgs: %s", (people, cids, orgs))
        print(scope_ids)

        rows = similar_chunks(q_vec, req.k, scope_ids) or similar_chunks(q_vec, req.k)
        if rows:

            chunks = build_context(rows)
            prompt = history_msgs + compose_prompt(req.question, chunks)

            chat = GROQ.chat.completions.create(model=CHAT_MODEL,
                                                messages=prompt, stream=False)
            answer = chat.choices[0].message.content.strip()

            # 2. persist user question & assistant answer
            save_turn(session_id, "user", req.question)
            save_turn(session_id, "assistant", answer)

            if not answer:
                answer = "No data available for this question."

            return AskResp(answer=answer, sources=[sid for sid, _ in chunks])
        else:
            return AskResp(answer="No data available for this question.", sources=[])
    except Exception as e:
        return AskResp(answer="Something went wrong.")

@app.get("/chunk/{chunk_id}")
def get_chunk(chunk_id: str):
    with psycopg2.connect(**PG) as pg, pg.cursor() as cur:
        cur.execute("SELECT text FROM documents WHERE chunk_id=%s", (chunk_id,))
        row = cur.fetchone()
        return {"text": row[0] if row else ""}