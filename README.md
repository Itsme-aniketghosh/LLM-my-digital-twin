# AI Digital Twin — Retrieval-Augmented Portfolio Assistant

A **production-style Retrieval-Augmented Generation (RAG) system** that builds an AI “digital twin” over personal documents (resume, statements, portfolio) and serves grounded, first-person responses through a conversational interface.

**Live deployment:**
🔗 [https://ag-hosh-my-digital-twin.hf.space/](https://ag-hosh-my-digital-twin.hf.space/)

---

## Overview

This project demonstrates how to design and deploy a **document-grounded LLM system** with explicit control over retrieval, chunking, context assembly, and generation parameters.

Rather than fine-tuning, the system uses **semantic retrieval + prompt-grounded generation** to ensure responses remain accurate, explainable, and aligned with source documents.

---

## System Architecture

```
User Query
   ↓
Dense Embedding
   ↓
FAISS Vector Retrieval
   ↓
Context Assembly
   ↓
Llama 3.1 Instruct
   ↓
Grounded Response
```

---

## Core Design

### Retrieval-Augmented Generation

* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
* **Vector Store**: FAISS (L2 distance)
* **LLM**: `meta-llama/Llama-3.1-8B-Instruct`
* **Interface**: Gradio (Hugging Face Spaces)

The LLM never answers without retrieved context. All responses are conditioned on document-derived evidence.

---

### Chunking Strategy

```python
CHUNK_SIZE = 1200       # words
CHUNK_OVERLAP = 200     # words
```

**Rationale**

* Personal documents are semantically dense and narrative-driven
* Larger chunks preserve continuity across experience sections
* Reduces fragmentation and improves reasoning quality
* Overlap prevents boundary loss between sections

---

### Retrieval & Context Assembly

* Top-K semantic retrieval using FAISS
* Similarity filtering to remove weak matches
* Deduplication of overlapping content
* Top results assembled into a single prompt context

This balances **precision, recall, and prompt budget efficiency**.

---

### Generation Configuration

```python
temperature = 0.8
top_p = 0.9
max_tokens = 700–900
```

* Optimized for natural but controlled responses
* First-person voice (“digital twin”)
* Factual, concise, and interview-oriented
---

## Performance (Observed)

* Vector search: sub-millisecond
* End-to-end latency: ~2–5 seconds
* Scales comfortably for small–medium document sets


---

## Technology Stack

* Python
* Sentence Transformers
* FAISS
* Llama 3.1
* Gradio
* Hugging Face Inference API


---

## Running Locally

```bash
pip install -r requirements.txt
python create_vector_db.py
python app.py
```

(Optional) Add a Hugging Face token for higher rate limits.


