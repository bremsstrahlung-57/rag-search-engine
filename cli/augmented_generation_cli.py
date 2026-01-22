import argparse

from lib.hybrid_search import HybridSearch
from lib.llm_funcs import (
    llm_question_answer,
    llm_summarize,
    llm_summary_with_citations,
    rag_output,
)
from lib.semantic_search import load_movies


def print_output(docs):
    for item in docs:
        title = item.get("doc").get("title")
        print(f"\t- {title}")


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results",
    )

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarizes Search results (search + summary)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results",
    )

    summary_cite_parser = subparsers.add_parser(
        "citations", help="Summarizes Search results with Sources (search + summary)"
    )
    summary_cite_parser.add_argument("query", type=str, help="Search query")
    summary_cite_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results",
    )

    question_parser = subparsers.add_parser("question", help="Ask Questions")
    question_parser.add_argument("query", type=str, help="Question")
    question_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results",
    )

    args = parser.parse_args()

    documents = load_movies()
    hyb = HybridSearch(documents)
    query = args.query
    limit = args.limit
    docs = hyb.rrf_search(query, limit=limit)

    match args.command:
        case "rag":
            rag_response = rag_output(query, docs)
            print("Search Result:")
            print_output(docs)
            print("\nRAG Response:")
            print(rag_response)

        case "summarize":
            summary = llm_summarize(query, docs)
            print("Search Result:")
            print_output(docs)
            print("\nLLM Summary:")
            print(summary)

        case "citations":
            summary_with_citations = llm_summary_with_citations(query, docs)
            print("Search Result:")
            print_output(docs)
            print("\nLLM Answer:")
            print(summary_with_citations)

        case "question":
            answer = llm_question_answer(query, docs)
            print("Search Result:")
            print_output(docs)
            print("\nAnswer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
