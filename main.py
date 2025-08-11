import os
# prevent transformers from importing torchvision (not needed here)
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# -----------------------------
# Model + tokenizer (LLaMA-arch)
# -----------------------------
MAXLEN = 1024
BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # llama-arch, gguf-friendly

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)

# ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map={"": device},
)

# make sure model knows pad id & disable KV cache during training
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# -----------------------------
# LoRA (LLaMA-style targets)
# -----------------------------
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
)
model = get_peft_model(model, lora_cfg)

# -----------------------------
# Dataset (Gretel text-to-sql)
# -----------------------------
ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")

def to_pair(ex):
    instr = (
        "You are an SQL generator. Using the database context below, "
        "write a valid SQL query that answers the user's question.\n\n"
        f"### Context:\n{ex['sql_context']}\n\n"
        f"### Question:\n{ex['sql_prompt']}"
    )
    out = (
            "<sql_query>\n" + ex["sql"] + "\n</sql_query>\n\n"
                                          "<explanation>\n" + str(ex.get("sql_explanation", "")) + "\n</explanation>"
    )
    return {"text": f"### Instruction:\n{instr}\n\n### Response:\n{out}"}

train = (
    ds.shuffle(seed=42)
    .select(range(5000))                 # faster run
    .map(to_pair, remove_columns=ds.column_names)
)

# -----------------------------
# Trainer config (no mixed precision on MPS)
# -----------------------------
OUTPUT_DIR = "tinyllama-sql-lora"

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=500,                 # quick PoC
    warmup_steps=20,
    logging_steps=10,
    save_steps=250,
    fp16=False, bf16=False,       # IMPORTANT: off for accelerate on MPS
    dataloader_pin_memory=False,
    optim="adamw_torch",          # <- avoid Accelerate optimizer shims
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train,
    args=args,
    dataset_text_field="text",
    max_seq_length=MAXLEN,
    packing=False,
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Merge LoRA and save full HF model
# -----------------------------
outdir = "merged-model-tinyllama"
if isinstance(model, PeftModel):
    merged = model.merge_and_unload()
    merged.save_pretrained(outdir, safe_serialization=True)
    tokenizer.save_pretrained(outdir)
else:
    model.save_pretrained(outdir, safe_serialization=True)
    tokenizer.save_pretrained(outdir)

print(f"Saved merged model to ./{outdir}")
