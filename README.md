# Fine-tuning a Small Llama Model with Ollama and LoRA

Workshop material for **PyCon Colombia 2026 (Medellín)**.

## Overview

This workshop teaches how to fine-tune a small Llama model using the **LoRA** (Low-Rank Adaptation) method and serve it locally with **Ollama**. The workshop runs on **Google Colab**, so attendees get free GPU access without any local setup. The running example is a **bank transaction intelligence** assistant: transaction categorization, fraud detection, spending insights, and customer support Q&A on synthetic data.

By the end of the workshop, attendees will understand how to:

- Prepare, validate, and explore an instruction dataset for fine-tuning.
- Fine-tune a small Llama model (TinyLlama-1.1B-Chat) with LoRA using `transformers`, `peft`, and `accelerate` on a Colab GPU runtime.
- Export the merged model to GGUF and serve it locally with Ollama.
- Evaluate the fine-tuned model against the base model on a held-out test set.

## Requirements

- A Google account (to run the notebooks on [Google Colab](https://colab.research.google.com/))
- [Ollama](https://ollama.com/) installed locally to serve the fine-tuned model
- [uv](https://docs.astral.sh/uv/) if running the notebooks locally instead of on Colab

## Setup

### Google Colab (recommended)

Open `notebooks/01_setup_environment.ipynb` on Colab first — it clones this repository, installs Ollama, and installs Python dependencies from `pyproject.toml` via `uv`. Attach a GPU runtime (`Runtime > Change runtime type > T4 GPU`) before running it. Run the remaining notebooks in order (`02` → `03` → `04`); each assumes the working directory and installed packages from `01` are already in place.

### Local (optional)

```bash
uv sync
uv run jupyter notebook
```

## Notebooks

Run in order — each depends on state (cloned repo, installed deps, cached model, trained/served model) left behind by the previous one:

1. **`01_setup_environment.ipynb`** — clones the repo, checks for GPU, installs Ollama, installs Python deps with `uv` from `pyproject.toml`, restarts the kernel (required so the freshly installed packages load cleanly), starts the Ollama service, and pre-downloads the TinyLlama base model.
2. **`02_data_exploration.ipynb`** — generates the synthetic bank transaction dataset (via `scripts/prepare_data.py`), validates it against the `TrainingExample` schema, and plots instruction/output length and task-type distributions.
3. **`03_finetuning.ipynb`** — LoRA fine-tunes TinyLlama on the training split (via `src/trainer.py`), plots the loss curve, exports the merged model to GGUF, registers it with Ollama, and sanity-checks the served model.
4. **`04_evaluation.ipynb`** — benchmarks the served fine-tuned model against the held-out test set and a set of out-of-distribution prompts, and reports similarity/quality metrics per task category.

## Repository structure

- `notebooks/` — the workshop's main hands-on material, designed to run on Colab in order (see above).
- `src/` — reusable Python modules imported by both notebooks and scripts:
  - `data_loader.py` — `TrainingExample` pydantic schema and `DataLoader` for loading/validating JSONL or CSV datasets.
  - `trainer.py` — `OllamaTrainer`, the LoRA fine-tuning pipeline (tokenization, training loop, adapter merge, GPU cleanup).
  - `ollama_export.py` — converts a merged Hugging Face model to GGUF (via llama.cpp) and registers it with Ollama through a generated `Modelfile`.
- `scripts/` — standalone CLI entry points that wrap the same `src/` logic the notebooks use inline, for local/scripted runs outside a notebook:
  - `prepare_data.py` — generates the synthetic bank transaction dataset and splits it into train/eval/test.
  - `finetune_llama.py` — runs `OllamaTrainer` from the command line (`--config`, `--model`, `--epochs`, `--batch-size` overrides).
  - `serve_model.py` — runs the GGUF export + Ollama registration from the command line, with an optional test prompt.
- `config/training_config.yaml` — LoRA hyperparameters, training args, data split ratios, and output/checkpoint settings consumed by `OllamaTrainer`.

## License

[MIT](LICENSE). See [CONTRIBUTORS.md](CONTRIBUTORS.md) for authorship.
