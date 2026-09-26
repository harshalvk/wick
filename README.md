# Wick

A from-scratch LLM inference server, built to understand how production serving engines (vLLM, TGI, llama.cpp) actually work under the hood.

## What this is

Wick implements an LLM inference pipeline from first principles — model loading, a hand-written transformer forward pass, KV caching, and a continuous batching scheduler — wrapped in an HTTP server. The goal is depth over convenience: most components are built manually rather than relying on `model.generate()`, in order to actually understand prefill/decode, attention internals, and request scheduling rather than treating them as a black box.

## Status

🚧 Early development — currently in **Phase 0** (model loading + manual forward pass, no server yet).

## Project phases

- [ ] **Phase 0** — Load model, implement transformer forward pass by hand, verify against Hugging Face reference
- [ ] **Phase 1** — Wrap in a basic HTTP server (single request at a time)
- [ ] **Phase 2** — Add KV caching
- [ ] **Phase 3** — Add request queue + static batching
- [ ] **Phase 4** — Upgrade to continuous batching with a scheduler
- [ ] **Phase 5** — Streaming responses + performance metrics

## Model

Currently targeting [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) for fast iteration.

## Setup

```bash
conda create -n wick python=3.11 -y
conda activate wick
pip install -r requirements.txt
python download_model.py
```

## Project structure
```bash
wick/
├── wick/
│ ├── model/ # hand-written transformer implementation
│ ├── server/ # HTTP serving layer (Phase 1+)
│ └── tests/ # correctness tests against HF reference
├── models/ # downloaded model weights (gitignored)
├── download_model.py
└── requirements.txt
```

## Running tests

```bash
python -m wick.tests.test_rmsnorm
python -m wick.tests.test_rope
```
