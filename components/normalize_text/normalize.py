import argparse
import os
import re
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Replace URLs
    text = re.sub(r"http\S+|www\S+", " <url> ", text)

    # Replace numbers
    text = re.sub(r"\d+", " <number> ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Trim
    text = text.strip()

    return text


def main():
    args = parse_args()

    # Load dataset
    input_path = os.path.join(args.data, "data.parquet")
    df = pd.read_parquet(input_path)

    # 🔍 DEBUG (you can remove later)
    print("DEBUG COLUMNS:", df.columns.tolist())
    print("DEBUG SHAPE:", df.shape)


    # Fix review text column
    if "reviewText_x" in df.columns:
        df["reviewText"] = df["reviewText_x"]
    elif "reviewText" not in df.columns:
        raise ValueError(f"No review text column found. Columns: {df.columns}")

    # Fix label column
    if "overall_x" in df.columns:
        df["overall"] = df["overall_x"]
    elif "overall" not in df.columns:
        raise ValueError(f"No overall column found. Columns: {df.columns}")


    text_column = "reviewText"

    df[text_column] = df[text_column].apply(normalize_text)

    df = df[df[text_column].str.len() >= 10]


    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(output_path)

    print("Rows after normalization:", len(df))


if __name__ == "__main__":
    main()