import json, pathlib, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ────────────────────────────────  USER CONFIG  ────────────────────────────────
BASE_MODEL   = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH    = "data/qa.jsonl"
OUTPUT_DIR   = "data/adapters"
MAX_LENGTH   = 512
EPOCHS       = 1
BATCH_SIZE   = 2
LR           = 5e-5
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
# ────────────────────────────────────────────────────────────────────────────────


def load_dataset(path: str) -> Dataset:
    """Load JSONL → HuggingFace Dataset and wrap into prompt-response format."""
    with open(path) as f:
        rows = [json.loads(l) for l in f]

    def _to_prompt(r):
        return {
            "text": (
                "### Instruction:\n" + r["instruction"].strip() +
                "\n\n### Response:\n"  + r["output"].strip()
            )
        }

    return Dataset.from_list([_to_prompt(r) for r in rows])


def tokenize(ds: Dataset, tokenizer) -> Dataset:
    return ds.map(
        lambda batch: tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        ),
        batched=True,
        remove_columns=["text"],
    )


def main() -> None:
    ds = load_dataset(DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_ds = tokenize(ds, tokenizer)

    # 8-bit quantised base weights (needs bitsandbytes)
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="cpu",
    )

    # Attach a fresh LoRA adapter
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_accumulation_steps=4,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds,
        data_collator=collator,
    )

    print("👉  Training started...")
    trainer.train()
    print("Training done, saving LoRA adapter to", OUTPUT_DIR)
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()

