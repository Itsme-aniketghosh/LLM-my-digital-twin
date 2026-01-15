# AI Digital Twin - RAG Portfolio Assistant

A retrieval-augmented generation (RAG) chatbot trained on personal documents (resume, SOPs, statements) using semantic search and Llama 3.1 for natural conversation.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add documents to my_files/
my_files/
├── resume.pdf
├── statement_of_purpose.txt
└── personal_statement.pdf

# 3. Create vector database
python create_vector_db.py

# 4. Add HF token (optional but recommended)
echo "HF_TOKEN=hf_your_token_here" > .env

# 5. Run
python app.py
```

## Technical Architecture

### RAG Pipeline

```
Query → Embedding → FAISS Search → Context Retrieval → LLM → Response
```

**Components:**
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector Store**: FAISS with L2 distance
- **LLM**: Llama 3.1 8B Instruct (via HuggingFace API)
- **Retrieval**: Semantic similarity with relevance filtering

### Chunking Strategy

```python
CHUNK_SIZE = 300 words
OVERLAP = 50 words
THRESHOLD = 0.25 similarity
TOP_K = 20 initial retrieval
CONTEXT_SIZE = 15 chunks (max ~4500 words)
```

**Why these parameters:**
- **300 words**: Optimal balance - enough context, granular retrieval
- **50 word overlap**: Maintains context continuity across chunks
- **0.25 threshold**: Inclusive filtering, captures related content
- **20 → 15 chunks**: Retrieve more, use best matches

### Retrieval Process

1. **Query Encoding**: Transform query to 384-dim vector
2. **FAISS Search**: L2 distance, retrieve 20 nearest neighbors
3. **Similarity Filtering**: Keep chunks with similarity > 0.25
4. **Deduplication**: Remove near-duplicate content (first 100 chars hash)
5. **Ranking**: Sort by similarity score
6. **Context Assembly**: Top 15 chunks → concatenate → send to LLM

### Similarity Scoring

```python
similarity = 1 / (1 + L2_distance)

# Score interpretation:
# 0.8-1.0 = Highly relevant
# 0.5-0.8 = Relevant  
# 0.3-0.5 = Somewhat relevant
# <0.3 = Filtered out
```

### LLM Integration

**Model**: `meta-llama/Llama-3.1-8B-Instruct`

**Parameters:**
```python
max_tokens = 700-900
temperature = 0.7  # Balanced creativity/consistency
top_p = 0.9
```

**System Prompt Strategy:**
- First-person responses (digital twin)
- Grounded in retrieved context
- Authentic tone, minimal fluff
- Entry-level focused for job analysis

## Features

### 1. Chat Interface
- Answers questions using retrieved context
- Responds as first-person digital twin
- Context-aware, conversational

### 2. Job Fit Analysis
- Semantic matching between job description and portfolio
- Realistic scoring (8-9+ for matching entry-level roles)
- Identifies strengths, gaps, and growth areas

## Technical Stack

```
sentence-transformers  # Dense embeddings
faiss-cpu             # Vector similarity search
PyPDF2                # PDF text extraction
gradio                # Web UI
huggingface_hub       # LLM inference API
python-dotenv         # Environment config
```

## Vector Database Structure

```
vector_db/
├── faiss_index.bin      # FAISS index (float32)
├── documents.pkl        # Chunk metadata + text
└── embeddings.npy       # Dense vectors (N x 384)
```

**Metadata per chunk:**
```python
{
    'text': str,          # Chunk content
    'source': str,        # Source filename
    'chunk_id': int,      # Position in document
    'type': str           # pdf/txt
}
```

## Performance

- **Embedding**: ~0.1s per query
- **FAISS Search**: <0.01s for 20 neighbors
- **LLM Generation**: 2-4s (API dependent)
- **Total Response Time**: 2-5s

## Key Design Decisions

### Why FAISS?
- Fast approximate nearest neighbor search
- Efficient L2 distance computation
- Low memory footprint
- Good for small-medium datasets (<100k vectors)

### Why 300-word chunks?
- Tested 200/300/400/600 word chunks
- 300 provides best precision/recall balance
- Smaller = more precise but fragmented context
- Larger = more context but diluted relevance

### Why 15 chunks in context?
- Llama 3.1 context window: 128k tokens
- 15 chunks ≈ 4500 words ≈ 6000 tokens
- Leaves room for prompt + response
- Comprehensive without overwhelming LLM

### Why similarity > 0.25?
- Empirically determined threshold
- 0.3 = too restrictive (misses relevant content)
- 0.2 = too loose (irrelevant chunks)
- 0.25 = sweet spot for recall without noise

## Optimization Tips

**For more precise retrieval:**
```python
CHUNK_SIZE = 250
THRESHOLD = 0.35
TOP_K = 15
```

**For more comprehensive context:**
```python
CHUNK_SIZE = 400
THRESHOLD = 0.20
TOP_K = 25
```

**For faster responses:**
```python
TOP_K = 10
CONTEXT_SIZE = 8
```

## Limitations

- FAISS uses exact L2 search (no ANN) - fine for <10k docs
- No query expansion or reranking
- Single-stage retrieval (no hybrid search)
- No caching of frequent queries
- Context window limited to 15 chunks

## Future Enhancements

- [ ] Hybrid search (BM25 + semantic)
- [ ] Cross-encoder reranking
- [ ] Query expansion with LLM
- [ ] Response caching
- [ ] Conversation memory
- [ ] Multi-turn context handling

---

**Built with:** Python 3.8+ | Sentence Transformers | FAISS | Llama 3.1 | Gradio
