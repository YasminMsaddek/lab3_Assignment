import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = os.path.join(args.data, "data.parquet")
    df = pd.read_parquet(input_path)

    text_column = "reviewText"

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found.")

    # Number of words
    df["review_length_words"] = df[text_column].apply(
        lambda x: len(str(x).split())
    )

    # Number of characters
    df["review_length_chars"] = df[text_column].apply(
        lambda x: len(str(x))
    )

    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(output_path)

    print("Added review length features.")


if __name__ == "__main__":
    main()