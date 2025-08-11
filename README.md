# SQL-LLM Fine-tuning & Deployment (TinyLLaMA + LoRA on MPS)

This repository shows how to **fine-tune a TinyLLaMA model** for **Text-to-SQL generation** using the [Gretel synthetic text-to-sql dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) with **LoRA adapters**, and then export it for **local inference in Ollama**.

This guide walks through training, converting, quantizing, and running a SQL-aware LLM locally with [llama.cpp](https://github.com/ggerganov/llama.cpp) and [Ollama](https://ollama.ai).

## 1. Environment Setup

### 1.1 Create & activate virtual environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 1.2 Install compatible dependencies
We pin to versions that work well together for training & conversion:

```bash
ip install --upgrade pip
pip uninstall -y transformers trl accelerate peft datasets huggingface_hub torchvision torchaudio

pip install \
    torch==2.4.1 \
    transformers==4.43.3 \
    trl==0.9.6 \
    accelerate==0.33.0 \
    peft==0.12.0 \
    datasets==2.20.0 \
    huggingface_hub==0.23.5
    
    ```

## 3. Train / Fine-tune Model

For faster runs:
- `MAXLEN=256`
- Use `select(range(500))` instead of full dataset
- `max_steps=20`

Run:
```bash
python main.py
```

## 4. Convert HuggingFace Model to GGUF

## 3. Convert to GGUF (for Ollama)

### 3.1 Clone llama.cpp
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

### 3.2 Convert merged HF model to GGUF
```bash
python3.12 convert_hf_to_gguf.py ../merged-model-tinyllama     --outfile ../llama-sql-f16.gguf     --outtype f16
```

---

## 4. Quantization (optional but recommended)

```bash
./build/bin/llama-quantize ../llama-sql-f16.gguf ../llama-sql-Q4_K_M.gguf Q4_K_M
```

Resulting file sizes:
```
llama-sql-f16.gguf  -> ~2.0 GB
llama-sql-Q4_K_M.gguf -> ~637 MB
```

## 7. Create Ollama Model

`Modelfile` example:
```
FROM ./llama-sql-Q4_K_M.gguf
TEMPLATE """
### Context:
{{{{ context }}}}

### Question:
{{{{ question }}}}

### Response:
"""
PARAMETER temperature 0
```

Create model in Ollama:
```bash
ollama create sql-llm -f Modelfile
```

## 8. Run Queries

Example:
```bash
ollama run sql-llm "Users(id INT, name TEXT, age INT)
Orders(id INT, user_id INT, total NUMERIC)

Question:
Find top 5 users by total spend."
```

Expected output:
```
<sql_query>
SELECT
  u.id,
  u.name,
  COALESCE(SUM(o.total), 0) AS total_spend
FROM Users u
LEFT JOIN Orders o ON o.user_id = u.id
GROUP BY u.id, u.name
ORDER BY total_spend DESC
LIMIT 5;
</sql_query>
<explanation>
Join Orders to Users on user_id, aggregate totals, sort by spend descending, and return top 5 users.
</explanation>
```

---
**Notes:**
- Ollama’s template variables (`{{{{ context }}}}`, `{{{{ question }}}}`) require a matching `TEMPLATE` in `Modelfile`.
- If Ollama fails with `unknown flag: --var`, pass all input as a single string instead of separate `--var` arguments.
