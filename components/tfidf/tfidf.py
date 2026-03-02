import argparse
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.train_data)

    # Initialize TF-IDF
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    
    # Fit and transform
    tfidf_matrix = tfidf.fit_transform(df['reviewText'].fillna(''))
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    
    # Keep the IDs for merging later
    final_df = pd.concat([df[['asin', 'reviewerID']].reset_index(drop=True), tfidf_df], axis=1)

    os.makedirs(args.output_data, exist_ok=True)
    final_df.to_parquet(os.path.join(args.output_data, "tfidf_features.parquet"))

if __name__ == "__main__":
    main()