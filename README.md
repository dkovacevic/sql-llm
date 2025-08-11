# SQL-LLM Fine-tuning & Deployment (TinyLLaMA + LoRA on MPS)

This repository shows how to **fine-tune a TinyLLaMA model** for **Text-to-SQL generation** using the [Gretel synthetic text-to-sql dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) with **LoRA adapters**, and then export it for **local inference in Ollama**.

It’s optimized for **Apple Silicon (M1/M2/M3)** using **MPS** acceleration.

---

## 1. Environment Setup

### 1.1 Create & activate virtual environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 1.2 Install compatible dependencies
We pin to versions that work well together for training & conversion:

```bash
pip install --upgrade pip
pip uninstall -y transformers trl accelerate peft datasets huggingface_hub torchvision torchaudio

pip install     torch==2.4.1     transformers==4.43.3     trl==0.9.6     accelerate==0.33.0     peft==0.12.0     datasets==2.20.0     huggingface_hub==0.23.5
```

> **Note:** We skip installing `torchvision` / `torchaudio` since they're not needed here and can cause MPS issues on macOS.

---

## 2. Training

### 2.1 Model choice
We use [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) because:
- LLaMA architecture → exportable to GGUF for Ollama.
- Small enough to train quickly on MPS.
- Compatible with LoRA adapters.


Run:
```bash
python3.12 main.py
```

---

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
./quantize ../llama-sql-f16.gguf ../llama-sql-q4.gguf q4_0
```

---

## 5. Create Ollama Model

### 5.1 Create a `Modelfile`
```
FROM ./llama-sql-q4.gguf
PARAMETER stop "<|im_end|>"
TEMPLATE """{{ .Prompt }}"""
```

### 5.2 Register with Ollama
```bash
ollama create sql-llm -f Modelfile
```

### 5.3 Run it
```bash
ollama run sql-llm
```
