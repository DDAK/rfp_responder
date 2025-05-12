
A fully **agent‑oriented** pipeline that turns an input directory of
text/PDF/DOCX files into a polished, metadata‑tagged JSONL dataset for LoRA
fine‑tuning.  Four collaborating agents (Generator, Judge, Rewriter, Metadata):

1. **Generator Agent** – produces a prompt/completion pair from each text chunk.
2. **Judge Agent** – scores the pair 0‑10 and either *ACCEPTS* or *RETRY*.
3. **Rewriter Agent** – if the pair scores **8 or 9** it polishes wording.
4. **Metadata Agent** – assigns a *domain* and *topic* label to every accepted
   pair for balanced sampling downstream.

Retries for rejected chunks are capped at **3**.

-------------
Usage example
-------------
For now supports **OpenAI *and* local Ollama models**.  Select backend via
`--backend openai|ollama` (default: *openai*).

``` ssh
# Run with local Ollama ⇣
python dataset.py \
  --backend ollama \
  --model mistral \
  --judge_model mistral \
  --rewrite_model mistral \
  --meta_model mistral \
  --input_dir ./docs --output_file dataset.jsonl
```

Behind the scenes we set the CrewAI agent’s LLM provider to **`ollama`** and
forward the chosen model name.  OpenAI usage is unchanged.

Fine tunening the model
finetune.py  –  LoRA adapter training for Llama-family models
────────────────────────────────────────────────────────────────────
• Works on CPU or GPU.
• Base default: meta-llama/Meta-Llama-3-8B-Instruct
• Only trains a tiny LoRA adapter (rank-16) ⇒ few-MB output.
• Expects a JSONL dataset:
      {"instruction": "...", "output": "..."}

```ssh
python finetune.py
```
Need to login to hugginface using huggingface-cli