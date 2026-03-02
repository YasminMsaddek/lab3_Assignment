import argparse
import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    args = parser.parse_args()

    # Download VADER lexicon
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()

    df = pd.read_parquet(args.input_data)

    def get_sentiment(text):
        if not text: return 0.0
        return sid.polarity_scores(text)['compound']

    df['sentiment_score'] = df['reviewText'].apply(get_sentiment)

    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(os.path.join(args.output_data, "sentiment_features.parquet"))

if __name__ == "__main__":
    main()