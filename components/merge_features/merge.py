import argparse
import os
import pandas as pd


KEYS = ["asin", "reviewerID", "overall"]


def resolve_parquet(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "data.parquet")
    return path


def load_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_parquet(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df_length = load_df(args.length)
    df_sentiment = load_df(args.sentiment)
    df_tfidf = load_df(args.tfidf)
    df_embeddings = load_df(args.embeddings)

    df = df_length.merge(df_sentiment, on=KEYS, how="inner")
    df = df.merge(df_tfidf, on=KEYS, how="inner")
    df = df.merge(df_embeddings, on=KEYS, how="inner")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(out_path, index=False)

    print("Merged rows:", len(df))
    print("Merged columns:", len(df.columns))
    print(f"Saved merged features to {out_path}")


if __name__ == "__main__":
    main()