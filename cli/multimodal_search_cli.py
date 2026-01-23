import argparse

from lib.multimodal_search import (
    MultimodalSearch,
    image_search_command,
    verify_image_embedding,
)


def main():
    parser = argparse.ArgumentParser(description="Rewrite a query based on an image")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify model embeddings"
    )
    verify_image_embedding_parser.add_argument(
        "image", type=str, help="Path to Image file"
    )

    image_search_parser = subparsers.add_parser("image_search", help="Search by Image")
    image_search_parser.add_argument("image_path", type=str, help="Path to Image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image_path
            verify_image_embedding(image_path=image_path)
        case "image_search":
            image_path = args.image_path
            out = image_search_command(image_path)
            for i, item in enumerate(out, start=1):
                title = item["title"]
                desc = item["description"][:100] + "..."
                score = item["similarity_score"]

                print(f"{i}. {title} (similarity: {score:.4f})")
                print(f"{desc}\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
