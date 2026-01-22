import argparse
import ast
import json
import logging
from time import sleep

from lib.hybrid_search import HybridSearch, normalize
from lib.llm_funcs import (
    batch_reranking,
    evaluate_result,
    groq_reranking,
    llm_query_enhance,
)
from lib.semantic_search import load_movies
from sentence_transformers import CrossEncoder

logging.basicConfig(
    filename="logs/cli.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def print_results(res):
    for i, item in enumerate(res, start=1):
        llm_score = item.get("llm_score", None)
        llm_rerank = item.get("llm_rerank", None)
        cross_encoder_score = item.get("cross_encoder_score", None)
        rrf_score = item.get("rrf_score")
        rrf_rank = item.get("rrf_rank")
        bm25_rank = item.get("bm_rank")
        sem_rank = item.get("sem_rank")
        title = item.get("doc", {}).get("title")
        desc = item.get("doc", {}).get("document") + "..."

        print(f"{i}. {title}")
        if llm_score is not None:
            print(f"Rerank Score: {llm_score}/10")
        elif llm_rerank is not None:
            print(f"Rerank Rank: {llm_rerank}")
        elif cross_encoder_score is not None:
            print(f"Cross Encoder Score: {cross_encoder_score:.4f}")
        else:
            print(f"RRF Rank: {rrf_rank}")
        print(f"RRF Score: {rrf_score:.4f}")
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

    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF search")
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
        choices=["individual", "batch", "cross_encoder"],
        help="LLM-based re-ranking",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="LLM-based result evaluation",
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

        case "rrf-search":
            if args.rerank_method:
                print(
                    f"Reranking top {args.limit} results using {args.rerank_method} method...\n"
                )
            print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")

            logging.info(f"Original Query: {args.query}")
            query = llm_query_enhance(args.query, args.enhance)
            logging.info(f"Enhanced Query: {query}")

            limit = 5 * args.limit if args.rerank_method else args.limit
            res = hyb.rrf_search(query, args.k, limit)

            if args.rerank_method == "individual":
                temp = res[: args.limit]
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
                logging.info(f"Final Result: {reranked}")
                print_results(reranked)

            elif args.rerank_method == "batch":
                temp = res[: args.limit]
                batch_rank = batch_reranking(
                    args.query,
                    temp,
                )
                lst = json.loads(batch_rank)
                ids_set = set(lst)
                for rank, item in enumerate(temp, start=1):
                    if item["id"] in ids_set:
                        item["llm_rerank"] = rank

                reranked = sorted(
                    temp,
                    key=lambda x: x["llm_rerank"],
                )
                logging.info(f"Final Result: {reranked}")
                print_results(reranked)

            elif args.rerank_method == "cross_encoder":
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                pairs = []
                for item in res:
                    title = item.get("doc", {}).get("title", "")
                    desc = item.get("doc", {}).get("document", "")
                    pairs.append(
                        [
                            query,
                            f"{title} - {desc}",
                        ]
                    )
                scores = cross_encoder.predict(pairs)

                for i, item in enumerate(res):
                    item["cross_encoder_score"] = scores[i]

                reranked = sorted(
                    res,
                    key=lambda x: x["cross_encoder_score"],
                    reverse=True,
                )
                logging.info(f"Final Result: {reranked}")
                print_results(reranked[: args.limit])

            else:
                print_results(res)

            if args.evaluate:
                eval = evaluate_result(args.query, res)
                eval_list = ast.literal_eval(eval)
                for item, ev in zip(res, eval_list):
                    item["eval"] = ev
                evaluated_res = sorted(
                    res,
                    key=lambda item: item["eval"],
                    reverse=True,
                )
                print("Evaluated Scores:")
                for i, item in enumerate(evaluated_res, start=1):
                    print(f"{i}. {item.get('doc', {}).get('title')}: {item['eval']}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
