import argparse
from time import sleep

from lib.hybrid_search import HybridSearch, normalize
from lib.llm_query import groq_reranking, llm_query_enhance
from lib.semantic_search import load_movies


def print_results(res):
    for i, item in enumerate(res, start=1):
        llm_score = item.get("llm_score", None)
        rrf_score = item.get("rrf_score")
        rrf_rank = item.get("rrf_rank")
        bm25_rank = item.get("bm_rank")
        sem_rank = item.get("sem_rank")
        title = item.get("doc", {}).get("title")
        desc = item.get("doc", {}).get("document") + "..."

        print(f"{i}. {title}")
        if llm_score is not None:
            print(f"Rerank Score: {llm_score}/10")
        else:
            pass
        print(f"RRF Rank: {rrf_rank}, RRF Score: {rrf_score:.4f}")
        print(f"BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
        print(f"{desc}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize given scores")
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="Scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted_search", help="Weighted search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=0.5,
        help="Alpha",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit",
    )

    rrf_search_parser = subparsers.add_parser("rrf_search", help="RRF search")
    rrf_search_parser.add_argument("query", type=str, help="Query")
    rrf_search_parser.add_argument(
        "--k",
        type=float,
        nargs="?",
        default=60,
        help="K - parameter",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual"],
        help="LLM-based re-ranking",
    )

    args = parser.parse_args()

    documents = load_movies()
    hyb = HybridSearch(documents)

    match args.command:
        case "normalize":
            res = normalize(args.scores)
            for score in res:
                print(f"* {score:.4f}")

        case "weighted_search":
            res = hyb.weighted_search(args.query, args.alpha, args.limit)

            for i, mov in enumerate(res, start=1):
                title = mov.get("doc").get("title")
                desc = mov.get("doc").get("description")
                bm25_score = mov.get("bm25_score")
                semantic_score = mov.get("semantic_score")
                hybrid_score = mov.get("hybrid_score")
                print(
                    f"{i}. {title}\nHybrid Score: {hybrid_score:.4f}\nBM25: {bm25_score:.4f}, Semantic: {semantic_score:.4f}\n{desc}\n"
                )

        case "rrf_search":
            query = llm_query_enhance(args.query, args.enhance)
            limit = 5 * args.limit if args.rerank_method else args.limit
            res = hyb.rrf_search(query, args.k, limit, args.rerank_method)

            if args.rerank_method == "individual":
                temp = res[:3]
                for i, item in enumerate(temp):
                    llm_score = groq_reranking(
                        query,
                        item.get("doc", {}).get("title"),
                        item.get("doc", {}).get("document") + "...",
                        item.get("rrf_rank"),
                        item.get("sem_rank"),
                        item.get("bm_rank"),
                    )
                    item["llm_score"] = llm_score
                    sleep(3)

                reranked = sorted(
                    temp,
                    key=lambda x: x["llm_score"],
                    reverse=True,
                )
                print_results(reranked)
            else:
                print_results(res)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
