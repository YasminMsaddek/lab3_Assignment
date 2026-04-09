import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Constant for columns that should not be used as features
DROP_COLS = {
    "asin", "reviewerID", "overall", "label",
    "reviewText", "reviewText_x", "reviewText_y",
    "title", "title_x", "title_y",
    "summary", "summary_x", "summary_y",
    "brand", "brand_x", "brand_y"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--solver", type=str, default='liblinear') 
    return parser.parse_args()

def process_split(path: str, sample_frac=1.0, ref_cols=None):
    """
    Loads Parquet data and processes list-columns (SBERT/TF-IDF) 
    efficiently to avoid SIGKILL.
    """
    target = os.path.join(path, "data.parquet") if os.path.isdir(path) else path
    df = pd.read_parquet(target)
    
    # Downsample to stay within RAM limits of the compute cluster
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    
    # Create binary label for sentiment
    df["label"] = (df["overall"] >= 4).astype(int)
    y = df["label"].values
    
    # Expand list columns (like SBERT embeddings) into individual columns
    for col in list(df.columns):
        if not df[col].empty and isinstance(df[col].iloc[0], (list, tuple)):
            print(f"Expanding feature column: {col}")
            expanded = pd.DataFrame(df[col].tolist(), index=df.index)
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            # Join and immediately delete the original list to save memory
            df = df.drop(columns=[col]).join(expanded)
            del expanded
            gc.collect() 
    
    # Drop non-numeric metadata
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=["number", "bool"]).astype(float).fillna(0)
    
    # Ensure consistency between Train/Val/Test sets
    if ref_cols is not None:
        for col in ref_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[list(ref_cols)]
    
    X = df.copy()
    f_names = X.columns.tolist()
    del df
    gc.collect()
    return X, y, f_names

def main():
    args = parse_args()
    
    # MLflow setup - Azure ML handles the Tracking URI automatically
    mlflow.sklearn.autolog() 
    
    with mlflow.start_run():
        print(f"Hyperparameters: C={args.C}, max_iter={args.max_iter}")

        # Use 30% sample to prevent Out of Memory on Standard_DS2_v2 / DS3_v2
        print("Processing Training data...")
        X_train, y_train, feature_columns = process_split(args.train_data, sample_frac=0.3)

        print("Processing Validation data...")
        X_val, y_val, _ = process_split(args.val_data, ref_cols=feature_columns)

        print(f"Training Logistic Regression with {len(feature_columns)} features...")
        model = LogisticRegression(
            C=args.C, 
            max_iter=args.max_iter, 
            solver=args.solver, 
            random_state=42
        )
        model.fit(X_train, y_train)

        # Log specific metrics for the Assignment requirements
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        mlflow.log_metric("val_accuracy", acc)
        print(f"Validation Accuracy: {acc}")
        
        # Save the model and the feature list to the output directory
        # This is required for Step VII (Deployment) to ensure score.py 
        # knows which columns to expect.
        os.makedirs(args.output, exist_ok=True)
        model_payload = {
            "model": model,
            "feature_columns": feature_columns
        }
        joblib.dump(model_payload, os.path.join(args.output, "model.pkl"))
        print(f"Model and feature metadata saved to {args.output}")

if __name__ == "__main__":
    main()