from __future__ import annotations

import argparse, json, os, textwrap
from pathlib import Path
from typing import List, Tuple, Dict
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

import  docx2txt, tiktoken
from tqdm import tqdm
from crewai import Agent, Task, Crew, Process

ENCODING = "cl100k_base"

def extract_text(path: Path) -> str:
    """Extract raw text from txt/md/pdf/docx; silently skip others."""
    suf = path.suffix.lower()
    try:
        if suf in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suf == ".pdf":
    	 # Load the document
            loader = PyMuPDF4LLMLoader(
        		        path,
        		        mode="single",
                    )
            docs = loader.load()
            print(docs[0].metadata)
            return docs[0].page_content
        if suf in {".docx", ".doc"}:
            return docx2txt.process(str(path))
    except Exception as e:
        print(f"[WARN] {path}: {e}")
    return ""

def chunk_text(text: str, max_tokens=2000, overlap=200) -> List[str]:
    enc = tiktoken.get_encoding(ENCODING)
    toks = enc.encode(text)
    out, start = [], 0
    while start < len(toks):
        end = min(start + max_tokens, len(toks))
        out.append(enc.decode(toks[start:end]))
        start = end - overlap
        if start < 0: start = 0
    return out

GEN_SYS = (
    "You are a knowledgeable assistant. From the provided CHUNK, create a "
    "question that requires the chunk to answer, then give the answer. Return "
    "JSON {\"prompt\": <q>, \"completion\": <a>} only."
)

JUDGE_SYS = textwrap.dedent("""
    You are a strict reviewer. Score 0‑10 for relevance, clarity & correctness.
    If score < {threshold} -> respond: RETRY <score>  ; else: ACCEPT <score>
""")

REWRITE_SYS = (
    "Stylistic editor: improve wording, grammar & conciseness of the JSON pair "
    "without altering facts. Output the revised JSON."
)

META_SYS = (
    "Taxonomy assistant: given the JSON pair output JSON {\"domain\": <area>, "
    "\"topic\": <topic>}."
)


def llm_dict(provider: str, model: str, temperature: float):
    """Return CrewAI‑compatible LLM dict for openai or ollama."""
    if provider == "openai":
        return {"model": model, "temperature": temperature}
    # provider == "ollama"
    return {"provider": "ollama", "model": model, "temperature": temperature}


def make_agent(role: str, backstory: str, sys_prompt: str,
               provider: str, model: str, temp: float) -> Agent:
    return Agent(
        role=role,
        goal=backstory,
        backstory=backstory,
        allow_delegation=False,
        llm=llm_dict(provider, model, temp),
        system_prompt=sys_prompt,
    )


def build_agents(models: Dict[str, str], provider: str, threshold: int):
    gen = make_agent("Generator", "Writes dataset pairs", GEN_SYS,
                     provider, models["gen"], 0.7)
    judge = make_agent("Judge", "Reviews dataset pairs",
                       JUDGE_SYS.format(threshold=threshold),
                       provider, models["judge"], 0.0)
    rewrite = make_agent("Rewriter", "Polishes accepted pairs", REWRITE_SYS,
                         provider, models["rewrite"], 0.3)
    meta = make_agent("Metadata", "Tags pairs", META_SYS,
                      provider, models["meta"], 0.0)
    return gen, judge, rewrite, meta

def run_pair_cycle(chunk: str, agents: Tuple[Agent, Agent, Agent, Agent],
                   rewrite_window=(8, 9), max_retries: int = 3):
    gen_agent, judge_agent, rewrite_agent, meta_agent = agents

    def run_task(desc: str, inp: Dict, agent: Agent, expect: str):
        task = Task(description=desc, inputs=inp, expected_output=expect, agent=agent)
        return Crew([agent], [task], process=Process.sequential).kickoff()[task.id]

    for _ in range(max_retries + 1):
        raw_gen = run_task("Generate Q/A JSON", {"CHUNK": chunk}, gen_agent, '{"prompt":')
        try:
            pair = json.loads(raw_gen)
        except json.JSONDecodeError:
            continue
        raw_judge = run_task("Judge pair", {"pair": json.dumps(pair)}, judge_agent, 'ACCEPT')
        try:
            label, score_s = raw_judge.split()
            score = int(score_s)
        except ValueError:
            continue
        if label == "RETRY":
            continue
        if rewrite_window[0] <= score <= rewrite_window[1]:
            raw_rewrite = run_task("Rewrite", {"pair": json.dumps(pair)}, rewrite_agent, '{"prompt":')
            try:
                pair = json.loads(raw_rewrite)
            except json.JSONDecodeError:
                pass
        raw_meta = run_task("Meta", {"pair": json.dumps(pair)}, meta_agent, '{"domain":')
        try:
            pair.update(json.loads(raw_meta))
        except json.JSONDecodeError:
            pass
        return pair
    return None

def build_dataset(input_dir: Path, output_path: Path, provider: str,
                  models: Dict[str, str], chunk_size: int, overlap: int,
                  judge_threshold: int):
    agents = build_agents(models, provider, judge_threshold)
    recs = []
    for p in tqdm(list(input_dir.rglob('*')), desc='Docs'):
        raw = extract_text(p)
        if not raw:
            continue
        for chk in chunk_text(raw, chunk_size, overlap):
            pair = run_pair_cycle(chk, agents, max_retries=3)
            if pair:
                recs.append(pair)
    with output_path.open('w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved {len(recs)} examples → {output_path}")


def parse_args():
    pa = argparse.ArgumentParser(description='Agentic LoRA dataset builder')
    pa.add_argument('--input_dir', required=True, type=Path)
    pa.add_argument('--output_file', required=True, type=Path)
    pa.add_argument('--backend', choices=['openai', 'ollama'], default='openai')
    pa.add_argument('--model', default='gpt-4o-mini')
    pa.add_argument('--judge_model', default='gpt-4o-mini')
    pa.add_argument('--rewrite_model', default='gpt-4o-mini')
    pa.add_argument('--meta_model', default='gpt-4o-mini')
    pa.add_argument('--chunk_size', type=int, default=2000)
    pa.add_argument('--overlap', type=int, default=200)
    pa.add_argument('--judge_threshold', type=int, default=8)
    pa.add_argument('--api_key', default=os.getenv('OPENAI_API_KEY'))
    return pa.parse_args()


def main():
    args = parse_args()

    if args.backend == 'openai' and not args.api_key:
        raise SystemExit('OPENAI_API_KEY missing')
    # CrewAI handles Ollama locally; no key needed.

    models = {
        'gen': args.model,
        'judge': args.judge_model,
        'rewrite': args.rewrite_model,
        'meta': args.meta_model,
    }

    build_dataset(args.input_dir, args.output_file, args.backend,
                  models, args.chunk_size, args.overlap, args.judge_threshold)

if __name__ == '__main__':
    main()
