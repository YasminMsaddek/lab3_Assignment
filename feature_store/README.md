# Lab 4: Text Feature Engineering with Azure ML

## Project Description
This lab implements an automated feature engineering pipeline in Azure ML. It takes the "Gold" data from Lab 3 and generates numerical features from raw Amazon review text.

## Features Extracted
1. **Text Normalization**: Standardized text by lowercasing and removing punctuation.
2. **Review Length**: Numerical features for word count and character count.
3. **Sentiment Analysis**: Used the VADER lexicon to calculate a compound sentiment score (-1 to 1).
4. **TF-IDF**: Created a 500-dimension vector representing word importance, fitted only on the training split to prevent data leakage.

## Pipeline Architecture
The pipeline consists of modular command components:
- **Splitter**: 70/15/15 Train/Val/Test split.
- **Extractors**: Parallel components for Length, Sentiment, and TF-IDF.
- **Merger**: Joins all features back into a single Parquet dataset.