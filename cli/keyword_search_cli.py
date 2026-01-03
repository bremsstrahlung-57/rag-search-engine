import argparse
import json
import math

from lib.keyword_search import InvertedIndex, process_text
from search_utils import BM25_B, BM25_K1


def load_data(filename):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
            return data["movies"]
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {filename}.")
        return []


Movies = load_data("data/movies.json")
index = {}
docmap = {}
term_freq = {}
inverted_index = InvertedIndex(index, docmap, term_freq)
(indexobj, docmapobj, tfobj, doclengthobj) = inverted_index.load()


def search_query(query):
    result = set()
    query_words = process_text(query)

    if indexobj is None:
        print("Error Loading Inverted Index")
        return []

    for q in query_words:
        try:
            result = result.union(indexobj[q])
        except KeyError:
            continue

    return sorted(result)


def print_result(res_list):
    for rank, (doc_id, score) in enumerate(res_list, start=1):
        print(f"{rank}. ({doc_id}) {docmapobj[doc_id]['title']} - Score: {score:.2f}")


def tf_func(doc_id, term):
    return inverted_index.get_tf(doc_id, term)


def idf_func(term):
    total_doc_count = len(docmapobj)
    term_match_doc_count = 0

    term = process_text(term)[0]

    for doc_id, tf_dict in tfobj.items():
        if tf_dict.get(term, 0) > 0:
            term_match_doc_count += 1

    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    return idf


def bm25_idf_command(term):
    bm25_score = inverted_index.get_bm25_idf(term)
    return bm25_score


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    bm25_tf_score = inverted_index.get_bm25_tf(doc_id, term, k1)
    return bm25_tf_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build a Inverted Index")

    tf_parser = subparsers.add_parser("tf", help="Term frequencies of given term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term Frequency query")

    idf_parser = subparsers.add_parser(
        "idf", help="Inverse Document Frequency of given term"
    )
    idf_parser.add_argument("term", type=str, help="Inverse Document Frequency query")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="term frequency–inverse document frequency of given term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument(
        "term", type=str, help="term frequency–inverse document frequency query"
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results from BM25 search",
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            mov = search_query(args.query)
            # print_result(mov)
            print(mov)
        case "build":
            inverted_index.build()
            inverted_index.save()
        case "tf":
            tf = tf_func(args.doc_id, args.term)
            print(tf)
        case "idf":
            idf = idf_func(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_score = tf_func(args.doc_id, args.term)
            idf_score = idf_func(args.term)
            tf_idf = tf_score * idf_score
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            results = inverted_index.bm25_search(args.query, args.limit)
            print_result(results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
