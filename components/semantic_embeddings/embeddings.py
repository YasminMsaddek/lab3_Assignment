import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def resolve_parquet(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "data.parquet")
    return path


def get_text_column(df: pd.DataFrame) -> str:
    for col in ["reviewText", "reviewText_x", "reviewText_y"]:
        if col in df.columns:
            return col
    raise RuntimeError(f"No review text column found. Columns: {list(df.columns)}")


def main():
    args = parse_args()

    df = pd.read_parquet(resolve_parquet(args.data))

    text_col = get_text_column(df)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df[text_col].fillna("").astype(str).tolist(),
        show_progress_bar=True
    )

    keep_cols = [c for c in ["asin", "reviewerID", "overall"] if c in df.columns]

    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"sbert_{i}" for i in range(len(embeddings[0]))]
    )

    out_df = pd.concat(
        [df[keep_cols].reset_index(drop=True), emb_df.reset_index(drop=True)],
        axis=1
    )

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Semantic embeddings created.")
    print("Output columns:", out_df.columns.tolist())
    print("Rows:", len(out_df))


if __name__ == "__main__":
    main()