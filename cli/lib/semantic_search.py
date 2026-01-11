import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer


def load_movies():
    with open("data/movies.json", "r") as file:
        documents = json.load(file)["movies"]
        return documents


def verify_model():
    search = SemanticSearch()
    model = search.model
    max_length = model.max_seq_length
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {max_length}")


def embed_text(text):
    embed_t = SemanticSearch()
    embedding = embed_t.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search = SemanticSearch()
    with open("data/movies.json", "r") as file:
        documents = json.load(file)["movies"]
        embeddings = search.load_or_create_embeddings(documents)
        print(f"Number of docs: {len(documents)}")
        print(
            f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
        )


def normal_chunking(text, chunk_size, overlap):
    words = text.split()
    chunks = []

    if overlap < 0:
        overlap = 0
    if len(words) < overlap:
        overlap = len(words)

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


def semantic_chunking(text, max_chunk_size, overlap):
    text_strip = text.strip()
    if text_strip == "":
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not sentences[0].endswith(("?", ".", "!")):
        words = text_strip.split()
        sentences = words

    if len(sentences) <= max_chunk_size:
        return [" ".join(sentences)] if sentences else []

    chunks = []
    step = max(1, max_chunk_size - overlap)

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + max_chunk_size])
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    search = SemanticSearch()
    embeddings = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embeddings[:5]}")
    print(f"Shape: {embeddings.shape}")


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embed_npy = "cache/movie_embeddings.npy"

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Text is Empty!")

        list_text = [text]
        embedding = self.model.encode(list_text)
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        movie_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embed_npy, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.embed_npy):
            self.embeddings = np.load(self.embed_npy)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        embedded_query = self.generate_embedding(query)
        tuples = []
        for i, vec in enumerate(self.embeddings):
            cos_sim_score = cosine_similarity(embedded_query, vec)
            tuples.append((cos_sim_score, self.documents[i]))

        sorted_list = sorted(tuples, key=lambda x: x[0], reverse=True)
        return sorted_list[:limit]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_npy = "cache/chunk_embeddings.npy"
        self.chunk_metadata_json = "cache/chunk_metadata.json"

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}

        all_chunks = []
        chunk_metadata = []

        for doc_idx, doc in enumerate(self.documents):
            description = doc.get("description", "")

            if not description.strip():
                continue

            chunk = semantic_chunking(description, 4, 1)
            total_chunks = len(chunk)

            for chunk_idx, chunk in enumerate(chunk):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save(self.chunk_embeddings_npy, self.chunk_embeddings)
        with open(self.chunk_metadata_json, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}
        if os.path.exists(self.chunk_embeddings_npy) and os.path.exists(
            self.chunk_metadata_json
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_npy)
            with open(self.chunk_metadata_json, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        text = self.generate_embedding(query)
        chunk_score = []

        for i, chunk in enumerate(self.chunk_embeddings):
            cosine_sim = cosine_similarity(chunk, text)
            metadata = self.chunk_metadata["chunks"][i]
            score = {
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "total_chunks": metadata["total_chunks"],
                "score": cosine_sim,
            }
            chunk_score.append(score)

        movie_idx_scores = {}
        for item in chunk_score:
            movie_id = item["movie_idx"]
            score = item["score"]
            if (
                movie_id not in movie_idx_scores
                or score > movie_idx_scores[movie_id]["score"]
            ):
                movie_idx_scores[movie_id] = item

        sorted_movies = sorted(
            movie_idx_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )
        top_movies = sorted_movies[:limit]

        results = []
        for movie_idx, item in top_movies:
            movie = self.document_map[movie_idx]
            data = {
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "score": round(item["score"], 4),
                "metadata": {
                    "chunk_idx": item["chunk_idx"],
                    "total_chunks": item["total_chunks"],
                },
            }
            results.append(data)
        return results
