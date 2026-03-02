import argparse
import os
import pandas as pd
import string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.input_data)

    # Normalize: Lowercase and remove punctuation
    def clean_text(text):
        if text:
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    df['reviewText'] = df['reviewText'].apply(clean_text)

    # Save output
    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(os.path.join(args.output_data, "normalized_data.parquet"))

if __name__ == "__main__":
    main()