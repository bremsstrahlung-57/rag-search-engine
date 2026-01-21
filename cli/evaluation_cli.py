import argparse

from lib.evaluation import load_dataset, rrf_test


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision @k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    test_cases = load_dataset("data/golden_dataset.json")
    rrf_test(test_cases, limit)


if __name__ == "__main__":
    main()
