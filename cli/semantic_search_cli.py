import argparse
import json

from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_query_text,
    embed_text,
    load_movies,
    normal_chunking,
    semantic_chunking,
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

    subparsers.add_parser("embed_chunks", help="Embedded Semantic chunking")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search from semantic chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Query to search")
    search_chunked_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit of results (default=5)",
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
            chunks = normal_chunking(args.text, args.chunk_size, args.overlap)

            print(f"Chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")

        case "semantic_chunk":
            chunks = semantic_chunking(args.text, args.max_chunk_size, args.overlap)

            print(f"Semantically chunking {len(args.text)} characters")
            for i, sentence in enumerate(chunks, start=1):
                print(f"{i}. {sentence}")

        case "embed_chunks":
            embed_sc = ChunkedSemanticSearch()
            with open("data/movies.json", "r") as file:
                documents = json.load(file)["movies"]
                embeddings = embed_sc.load_or_create_chunk_embeddings(documents)
                print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            movies = load_movies()
            search = ChunkedSemanticSearch()
            embeddings = search.load_or_create_chunk_embeddings(movies)
            result = search.search_chunks(args.query, args.limit)

            for i, data in enumerate(result, start=1):
                title = data["title"]
                score = data["score"]
                description = data["document"]
                print(f"\n{i}. {title} (score: {score:.4f})")
                print(f"   {description}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
