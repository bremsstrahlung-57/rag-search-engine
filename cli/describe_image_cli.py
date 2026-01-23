import argparse

from lib.describe_image import read_img


def main():
    parser = argparse.ArgumentParser(description="Rewrite a query based on an image")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file",
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()

    image_path = args.image
    query_text = args.query
    print("Image:", image_path)
    print("Query:", query_text)

    read_img(query=query_text, img=image_path)


if __name__ == "__main__":
    main()
