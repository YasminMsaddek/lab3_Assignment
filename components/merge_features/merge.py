import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment_data", type=str, required=True)
    parser.add_argument("--length_data", type=str, required=True)
    parser.add_argument("--tfidf_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    args = parser.parse_args()

    # Load all parts
    df_sent = pd.read_parquet(args.sentiment_data)
    df_len = pd.read_parquet(args.length_data)
    df_tfidf = pd.read_parquet(args.tfidf_data)

    # Join on keys
    # Note: We drop duplicate columns (like reviewText) from secondary dataframes before joining
    merged_df = df_sent.merge(df_len.drop(columns=['reviewText']), on=['asin', 'reviewerID'])
    merged_df = merged_df.merge(df_tfidf, on=['asin', 'reviewerID'])

    os.makedirs(args.output_data, exist_ok=True)
    merged_df.to_parquet(os.path.join(args.output_data, "final_features.parquet"))

if __name__ == "__main__":
    main()