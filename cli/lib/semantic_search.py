import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


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
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
