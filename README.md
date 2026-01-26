# RAG Search Engine

A full-featured Retrieval Augmented Generation (RAG) search engine I built while taking the [Boot.dev](https://boot.dev) RAG course. This project is a deep dive into modern search techniques — from classic keyword matching to semantic understanding, and finally to LLM-powered answer generation.

The idea is simple: take a movie database and build a search engine that actually *understands* what you're looking for, not just matching keywords.

## What I Built

### Keyword Search (BM25)

Started with the basics — an inverted index that maps terms to documents. But plain term frequency isn't enough, so I implemented the full BM25 algorithm:

- **Inverted Index**: Fast lookups for which documents contain which terms
- **TF-IDF**: Classic term frequency × inverse document frequency scoring
- **BM25**: The industry-standard ranking function with tunable `k1` (saturation) and `b` (length normalization) parameters
- **Text Processing**: Tokenization, Porter stemming, stopword removal using NLTK

```sh
# Build the index
python keyword_search_cli.py build

# Search with BM25
python keyword_search_cli.py bm25search "space adventure"
```

### Semantic Search

Keywords only get you so far. Sometimes you want to search for "movies about friendship" and get results even if they don't contain the word "friendship". That's where embeddings come in.

- **Sentence Transformers**: Using `all-MiniLM-L6-v2` to generate 384-dimensional embeddings
- **Cosine Similarity**: Finding semantically similar documents
- **Chunking**: Breaking long descriptions into smaller pieces for better matching
  - Normal chunking (fixed window with overlap)
  - Semantic chunking (sentence-aware splitting)
- **Caching**: Embeddings are saved to disk so you don't have to regenerate them

```sh
# Run a semantic search
python semantic_search_cli.py search "heartwarming story about growing up"

# Search using chunked embeddings (more accurate for long descriptions)
python semantic_search_cli.py search_chunked "revenge thriller"
```

### Hybrid Search

Why choose between keyword and semantic search when you can have both? I implemented two fusion strategies:

- **Weighted Search**: Combine normalized BM25 and semantic scores with an alpha parameter
- **Reciprocal Rank Fusion (RRF)**: A rank-based fusion that's less sensitive to score calibration

```sh
# Weighted hybrid search
python hybrid_search_cli.py weighted_search "romantic comedy in paris" --alpha 0.6

# RRF search (generally more robust)
python hybrid_search_cli.py rrf-search "heist movie with twist ending"
```

### Multimodal Search

This is where it gets fun. Want to find movies similar to an image? Using CLIP (`clip-ViT-B-32`), the engine can embed both images and text into the same vector space.

```sh
# Search by image
python multimodal_search_cli.py image_search path/to/image.jpg
```

There's also a feature to rewrite text queries based on images using Gemini's vision capabilities:

```sh
python describe_image_cli.py --image poster.jpg --query "movies like this"
```

### Query Enhancement

Real users don't type perfect queries. I added LLM-powered query enhancement:

- **Spell Correction**: Fix typos without changing correct words
- **Query Rewriting**: Transform vague queries into specific, searchable terms
- **Query Expansion**: Add synonyms and related concepts

```sh
# RRF search with query rewriting
python hybrid_search_cli.py rrf-search "that bear movie with leo" --enhance rewrite
```

### Re-ranking

Initial retrieval gets you candidates, but re-ranking picks the winners. I implemented three approaches:

- **Individual LLM Scoring**: Ask an LLM to rate each result's relevance (0-10)
- **Batch LLM Ranking**: Give the LLM all results and ask it to order them
- **Cross-Encoder**: Use a specialized neural model (`ms-marco-TinyBERT-L2-v2`) for fast, accurate re-ranking

```sh
# Re-rank using cross-encoder (fast and good)
python hybrid_search_cli.py rrf-search "sci-fi movie" --rerank-method cross_encoder

# Re-rank using LLM (slower but can understand nuance)
python hybrid_search_cli.py rrf-search "sci-fi movie" --rerank-method batch
```

### Retrieval Augmented Generation

The whole point of RAG — use retrieved documents to generate better answers:

- **RAG Response**: Direct question answering based on retrieved movies
- **Summarization**: Synthesize information across multiple results
- **Citations**: Summarize with source attribution
- **Q&A**: Casual, conversational answers

```sh
# Get a RAG-powered answer
python augmented_generation_cli.py rag "what are some good movies about time travel?"

# Get a summary with citations
python augmented_generation_cli.py citations "recommend horror movies"
```

### Evaluation

You can't improve what you don't measure. I built an evaluation system using a golden dataset:

- **Precision@k**: Of the k results retrieved, how many were relevant?
- **Recall@k**: Of all relevant documents, how many did we retrieve?
- **F1 Score**: Harmonic mean of precision and recall

```sh
python evaluation_cli.py --limit 5
```

## Tech Stack

- **Python 3.14+** (using modern pattern matching)
- **Sentence Transformers** for embeddings
- **NLTK** for text processing
- **NumPy** for vector operations
- **Pillow** for image handling
- **Google Gemini** for LLM features (query enhancement, re-ranking, RAG)
- **Groq** for fast LLM inference (re-ranking)

## Setup

1. Clone the repo and set up the environment:

```sh
uv venv
uv sync
```

2. Create a `.env` file with your API keys:

```
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
MODEL=gemini-2.5-flash
GROQ_MODEL=llama-3.3-70b-versatile
```

3. Add your movie data to `data/movies.json` (expected format: `{"movies": [{"id": 1, "title": "...", "description": "..."}, ...]}`)

4. Run any of the CLI tools from the `cli/` directory.

## Project Structure

```
├── cli/
│   ├── lib/
│   │   ├── describe_image.py     # Multimodal query rewriting
│   │   ├── evaluation.py         # Precision, recall, F1
│   │   ├── hybrid_search.py      # Weighted + RRF fusion
│   │   ├── keyword_search.py     # Inverted index, BM25
│   │   ├── llm_funcs.py          # All LLM interactions
│   │   ├── multimodal_search.py  # CLIP-based image search
│   │   └── semantic_search.py    # Embeddings, chunking
│   ├── augmented_generation_cli.py
│   ├── describe_image_cli.py
│   ├── evaluation_cli.py
│   ├── hybrid_search_cli.py
│   ├── keyword_search_cli.py
│   ├── multimodal_search_cli.py
│   └── semantic_search_cli.py
├── cache/                         # Cached embeddings and indices
├── data/                          # Movie dataset and golden test cases
└── logs/                          # CLI logs
```

## What I Learned

This project was a great way to understand how modern search actually works under the hood.

- BM25 is still very effective, keyword matching catches things embeddings miss.
- Reciprocal Rank Fusion works well without needing score tuning.
- Re-ranking improves results significantly, though it is computationally expensive.
- LLMs are flexible and good for understanding queries, but slow for scoring every result.

---
