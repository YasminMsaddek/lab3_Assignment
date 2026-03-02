import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_data)

    # Character count
    df['review_len_char'] = df['reviewText'].str.len()
    # Word count
    df['review_len_word'] = df['reviewText'].str.split().str.len()

    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(os.path.join(args.output_data, "length_features.parquet"))

if __name__ == "__main__":
    main()