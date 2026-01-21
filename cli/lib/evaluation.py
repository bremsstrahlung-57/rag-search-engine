import json

from lib.hybrid_search import HybridSearch
from lib.reranking import evaluate_result
from lib.semantic_search import load_movies

documents = load_movies()
HYB = HybridSearch(documents)


def load_dataset(file):
    with open(file, "r") as f:
        data = json.load(f)
        return data["test_cases"]


def rrf_test(tests: list, limit: int):
    print(f"k = {limit}\n")

    for test in tests:
        query = test.get("query")
        relevant_data = test.get("relevant_docs")
        results = HYB.rrf_search(query=query, k=60, limit=limit)
        retrieved_data = [item.get("doc", {}).get("title") for item in results]
        total_relevant = len(relevant_data)
        total_retrieved = len(retrieved_data)

        prec = precision(retrieved_data, relevant_data, total_retrieved)
        recl = recall(retrieved_data, relevant_data, total_relevant)
        f1 = f1_score(prec, recl)

        print(f"- Query: {query}\n")
        print(f"- Precision@{limit}: {prec:.4f}")
        print(f"- Recall@{limit}: {recl:.4f}")
        print(f"- F1 Score: {f1:.4f}")
        print(f"- Retrieved: {', '.join(retrieved_data)}")
        print(f"- Relevant: {', '.join(relevant_data)}")
        print("\n")


def precision(retrieved_data: list, relevant_data: list, total_retrieved: int):
    relevant_retrieved = 0

    for mov in retrieved_data:
        if mov in relevant_data:
            relevant_retrieved += 1

    prec = relevant_retrieved / total_retrieved
    return prec


def recall(retrieved_data: list, relevant_data: list, total_relevant: int):
    relevant_retrieved = 0

    for rel in relevant_data:
        if rel in retrieved_data:
            relevant_retrieved += 1

    rec = relevant_retrieved / total_relevant
    return rec


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)
