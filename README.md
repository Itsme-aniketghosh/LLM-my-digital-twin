# AI Digital Twin — Knowledge-Grounded Portfolio Assistant

An AI that answers as **Aniket Ghosh**: an AI/ML engineer and researcher (ML engineer at Varosync, M.S. AI at Northeastern) working on AI safety — mechanistic interpretability and evaluations. It replies in the first person, grounded in a personal knowledge base, through a chat interface plus three job-targeted tools.

**Live demo:** [ag-hosh-my-digital-twin.hf.space](https://ag-hosh-my-digital-twin.hf.space/)

---

## Overview

A Retrieval-Augmented Generation (RAG) system that grounds an LLM in a personal knowledge base instead of fine-tuning. Every answer is conditioned on retrieved source text, so it stays close to what's actually written about me.

Knowledge lives in editable Markdown under [`knowledge/`](knowledge/). At startup the files are split by section, embedded, and indexed in memory. Each query pulls the most similar sections, which are combined with a fixed resume context that's always included.

---

## Features

- **Chat With Me.** Ask about my background, projects, interpretability/safety work, or how I work.
- **Job Fit Analysis.** Paste a job description and get an honest, specific read on fit, gaps included.
- **Cover Letter Generator.** Targeted, no-filler cover letters in my voice.
- **How I Can Help You.** A concrete value pitch for a given company or problem.

---

## System architecture

```
User Query
   ↓
Dense Embedding (all-MiniLM-L6-v2)
   ↓
Cosine Similarity Retrieval over knowledge/*.md
   ↓
Context Assembly (retrieved sections + always-on resume context)
   ↓
LLM (model fallback chain, streamed)
   ↓
Grounded First-Person Response
```

---

## How it works

**Knowledge and retrieval.** The Markdown in `knowledge/` is chunked by level-2 heading, embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, normalized), and held in memory. Retrieval is plain cosine similarity — top-K above a threshold, no external vector store. The retrieved sections sit next to a fixed resume context, so answers stay anchored even when retrieval misses.

**Generation with fallback.** The app walks a chain of models until one streams a real answer:

1. OpenRouter free-tier models first, with `openai/gpt-oss-120b:free` as primary and several alternates behind it.
2. `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face as the final fallback.

Answers stream token by token, and a half-streamed answer is never restarted if a later model in the chain fails.

---

## Configuration

The app picks a provider based on what you set:

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Use OpenRouter free-tier models (the preferred primary). |
| `HF_TOKEN` | Hugging Face Inference token, used for the fallback model. |
| `TWIN_MODEL` | Optional. Override the primary model id. |

Copy `.env.example` to `.env` and fill in whichever keys you have. To change what the twin knows, edit the files in `knowledge/`.

---

## Technology stack

Python, Sentence Transformers for embeddings, OpenRouter and the Hugging Face Inference API for generation, Gradio for the interface, NumPy for the math.

---

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Set `OPENROUTER_API_KEY` and/or `HF_TOKEN` in a `.env` file first (see [Configuration](#configuration)).
