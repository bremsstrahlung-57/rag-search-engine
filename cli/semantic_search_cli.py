import argparse
import json

from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify_model", help="Verify Model")
    subparsers.add_parser("verify_embeddings", help="Verify Embeddings")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed Text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed Query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using Semantic Search"
    )
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results from Semantic search",
    )

    args = parser.parse_args()

    match args.command:
        case "verify_model":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_s = SemanticSearch()
            with open("data/movies.json", "r") as file:
                documents = json.load(file)["movies"]
                search_s.load_or_create_embeddings(documents)
            result = search_s.search(args.query, args.limit)

            for i, (score, movie) in enumerate(result, start=1):
                print(
                    f"{i}. {movie['title']}: (score: {score:.4f})\n{movie['description'][:100]}..."
                )
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
