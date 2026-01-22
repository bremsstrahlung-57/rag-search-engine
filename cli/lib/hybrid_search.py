import logging
import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

logging.basicConfig(
    filename="logs/cli.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def normalize(scores: list[float | int]):
    max_score, min_score = max(scores), min(scores)
    if max_score == min_score:
        return [1 for _ in range(len(scores))]

    divisor = max_score - min_score
    normalized_scores = []

    for score in scores:
        normalized_score = (score - min_score) / divisor
        normalized_scores.append(normalized_score)

    return normalized_scores


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.index = {}
        self.docmap = {}
        self.term_freq = {}

        self.idx = InvertedIndex(self.index, self.docmap, self.term_freq)
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5, x=500):
        bm25_result = self._bm25_search(query, limit * x)
        semantic_result = self.semantic_search.search_chunks(query, limit * x)

        bm25_map = {doc_id: score for doc_id, score in bm25_result}
        semantic_map = {item["id"]: item["score"] for item in semantic_result}

        norm_bm25 = normalize(list(bm25_map.values()))
        norm_sem = normalize(list(semantic_map.values()))

        bm25_norm_map = dict(zip(bm25_map.keys(), norm_bm25))
        semantic_norm_map = dict(zip(semantic_map.keys(), norm_sem))

        common_ids = set(bm25_norm_map) & set(semantic_norm_map)
        semantic_doc_map = {d["id"]: d for d in semantic_result}

        scores = []
        for doc_id in common_ids:
            norm_bm25_score = bm25_norm_map[doc_id]
            norm_sem_score = semantic_norm_map[doc_id]

            hyb_score = hybrid_score(norm_bm25_score, norm_sem_score, alpha)

            sem_doc = semantic_doc_map[doc_id]

            scores.append(
                {
                    "id": doc_id,
                    "doc": {
                        "id": sem_doc["id"],
                        "title": sem_doc.get("title"),
                        "description": sem_doc.get("document"),
                    },
                    "bm25_score": norm_bm25_score,
                    "semantic_score": norm_sem_score,
                    "hybrid_score": hyb_score,
                }
            )

        scores.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return scores[:limit]

    def rrf_search(self, query, k=60, limit=5, x=500):
        bm25_result = self._bm25_search(query, limit * x)
        semantic_result = self.semantic_search.search_chunks(query, limit * x)

        results = {}
        for rank, doc in enumerate(semantic_result, start=1):
            doc_id = doc["id"]
            score = rrf_score(rank)

            if doc_id not in results:
                results[doc_id] = {
                    "id": doc_id,
                    "doc": doc,
                    "sem_rank": None,
                    "bm_rank": None,
                    "rrf_score": 0.0,
                }

            results[doc_id]["sem_rank"] = rank
            results[doc_id]["rrf_score"] += score

        for rank, doc in enumerate(bm25_result, start=1):
            doc_id = doc[0]
            score = rrf_score(rank)

            if doc_id not in results:
                results[doc_id] = {
                    "id": doc_id,
                    "doc": self.semantic_search.document_map[doc_id - 1],
                    "bm_rank": None,
                    "sem_rank": None,
                    "rrf_score": 0.0,
                }

            results[doc_id]["bm_rank"] = rank
            results[doc_id]["rrf_score"] += score

        fused = sorted(results.values(), key=lambda x: x["rrf_score"], reverse=True)[
            :limit
        ]

        for rank, item in enumerate(fused, start=1):
            item["rrf_rank"] = rank

        logging.info(f"RRF Search Result: {fused}")
        return fused
