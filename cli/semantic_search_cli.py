import argparse
import json
import re

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

    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunking to split long text into smaller pieces for embedding."
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "chunk_size",
        type=int,
        nargs="?",
        default=200,
        help="Size of chunk (default=200)",
    )
    chunk_parser.add_argument(
        "overlap",
        type=int,
        nargs="?",
        default=0,
        help="Overlap chunks",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantic chunking"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "max_chunk_size",
        type=int,
        nargs="?",
        default=4,
        help="Size of chunk (default=4)",
    )
    semantic_chunk_parser.add_argument(
        "overlap",
        type=int,
        nargs="?",
        default=0,
        help="Overlap chunks",
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

        case "chunk":
            words = args.text.split()
            chunks = []

            if args.overlap < 0:
                args.overlap = 0
            if len(words) < args.overlap:
                args.overlap = len(words)

            for i in range(0, len(words), args.chunk_size - args.overlap):
                chunk = " ".join(words[i : i + args.chunk_size])
                chunks.append(chunk)

            print(f"Chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")

        case "semantic_chunk":
            sentences = re.split(r"(?<=[.!?])\s+", args.text)
            chunks = []
            step = args.max_chunk_size - args.overlap
            i = 0

            while i + args.max_chunk_size <= len(sentences):
                chunk = " ".join(sentences[i : i + args.max_chunk_size])
                chunks.append(chunk)
                i += step

            print(f"Semantically chunking {len(args.text)} characters")
            for i, sentence in enumerate(chunks, start=1):
                print(f"{i}. {sentence}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
