import argparse
import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Download VADER lexicon (safe if already downloaded)
    nltk.download("vader_lexicon")

    sia = SentimentIntensityAnalyzer()

    input_path = os.path.join(args.data, "data.parquet")
    df = pd.read_parquet(input_path)

    text_column = "reviewText"

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found.")

    def get_sentiment(text):
        scores = sia.polarity_scores(str(text))
        return pd.Series([
            scores["pos"],
            scores["neg"],
            scores["neu"],
            scores["compound"]
        ])

    df[[
        "sentiment_pos",
        "sentiment_neg",
        "sentiment_neu",
        "sentiment_compound"
    ]] = df[text_column].apply(get_sentiment)

    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(output_path)

    print("Added sentiment features.")


if __name__ == "__main__":
    main()