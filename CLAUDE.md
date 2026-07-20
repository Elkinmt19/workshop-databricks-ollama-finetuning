# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Workshop material for PyCon Colombia 2026 (Medellín): teaches how to fine-tune a small Llama model using **Ollama** and the **LoRA** method. The workshop runs on **Google Colab** (free GPU runtime, no local setup for attendees); local execution is a secondary path for authoring/testing content.

## Current state

The four workshop notebooks, `src/`, `scripts/`, and `config/training_config.yaml` are populated and working end-to-end on Colab (setup → data exploration → LoRA fine-tuning → evaluation). There is still no test suite or lint/formatter config — don't assume `pytest`/`ruff`/`black` exist until they're added to `pyproject.toml`.

## Environment & commands

Dependency management is **uv** (`[tool.uv] package = false` — this is not a distributable package, just a workshop environment). Python `>=3.12`, standard PEP 621 `[project]` metadata (no Poetry).

```bash
uv sync                # install all deps (see pyproject.toml)
uv run jupyter notebook   # or: uv run jupyter lab
```

On **Google Colab** (the primary target platform), notebook `01_setup_environment.ipynb` installs Python dependencies via `uv pip install --system` against this repo's `pyproject.toml` — `uv`/`pyproject.toml` is the single source of truth for deps everywhere, including Colab. That install step deliberately skips `jupyter`/`notebook`/`ipywidgets`: Colab's kernel process already *is* that stack, and reinstalling those packages live overwrites C extensions the running kernel has mapped in memory, crashing it. The same cell restarts the kernel afterward via `ipykernel`'s `do_shutdown(restart=True)` (not a raw `os.kill`) so numpy and other C-extension deps upgraded on disk get loaded fresh rather than raising `ImportError: cannot load module more than once per process`.

## Stack

- **Ollama**: `ollama` Python client — for running/serving the fine-tuned small Llama model locally.
- **Fine-tuning**: `torch`, `transformers`, `peft`, `accelerate` — LoRA fine-tuning stack, run on Colab's GPU runtime.
- **Config/validation**: `pydantic`, `pydantic-settings`, `python-dotenv`.
- **Notebook/data**: `jupyter`, `notebook`, `ipywidgets`, `pandas`, `numpy`, `matplotlib`.

## Structure

- `notebooks/` — the primary teaching artifact, run in order: `01_setup_environment` (clone repo, install deps, install/start Ollama, cache base model) → `02_data_exploration` (generate + validate synthetic bank-transaction dataset) → `03_finetuning` (LoRA fine-tune, GGUF export, Ollama registration) → `04_evaluation` (benchmark served model vs. held-out test set + OOD prompts). Each notebook depends on state left by the previous one in the same runtime.
- `src/` — reusable Python modules imported by both notebooks and scripts: `data_loader.py` (`TrainingExample` pydantic schema, JSONL/CSV loading), `trainer.py` (`OllamaTrainer`, the LoRA fine-tuning pipeline), `ollama_export.py` (GGUF conversion via llama.cpp + `ollama create` registration).
- `scripts/` — standalone CLI entry points wrapping the same `src/` logic the notebooks use inline (`prepare_data.py`, `finetune_llama.py`, `serve_model.py`); not invoked by the notebooks themselves.
- `config/training_config.yaml` — LoRA hyperparameters, training args, data split ratios, and checkpoint/output settings consumed by `OllamaTrainer`.
