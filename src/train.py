import argparse
import os
import time
import joblib
import mlflow
import pandas as pd
import gc  # Added for memory management

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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

def load_data(path: str) -> pd.DataFrame:
    # Resolve parquet path if it's a directory
    target = os.path.join(path, "data.parquet") if os.path.isdir(path) else path
    return pd.read_parquet(target)

def process_split(path: str, ref_cols=None):
    """Loads, labels, and expands features for a split, then returns X and y."""
    df = load_data(path)
    df["label"] = (df["overall"] >= 4).astype(int)
    y = df["label"].values
    
    # Expand list columns
    for col in list(df.columns):
        series = df[col]
        non_null = series.dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], (list, tuple)):
            expanded = pd.DataFrame(series.tolist(), index=df.index)
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            df = df.drop(columns=[col]).join(expanded)
    
    # Drop text and select numeric
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=["number", "bool"]).astype(float).fillna(0)
    
    # Align columns if reference columns provided
    if ref_cols is not None:
        for col in ref_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[list(ref_cols)]
    
    X = df.copy()
    feature_names = X.columns.tolist()
    
    # CRITICAL: Clean up the intermediate dataframe
    del df
    gc.collect()
    
    return X, y, feature_names

def main():
    args = parse_args()
    with mlflow.start_run():
        mlflow.log_param("C", args.C)
        mlflow.log_param("solver", args.solver)

        print("Processing Training data...")
        X_train, y_train, feature_columns = process_split(args.train_data)

        print("Processing Validation data...")
        X_val, y_val, _ = process_split(args.val_data, ref_cols=feature_columns)

        print("Processing Test data...")
        X_test, y_test, _ = process_split(args.test_data, ref_cols=feature_columns)

        print(f"Training Logistic Regression (C={args.C}, solver={args.solver})...")
        model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver=args.solver, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate and log
        for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            acc = accuracy_score(y, model.predict(X))
            mlflow.log_metric(f"{name}_accuracy", acc)
            print(f"{name}_accuracy: {acc}")

        # Save
        os.makedirs(args.output, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": feature_columns}, os.path.join(args.output, "model.pkl"))
        mlflow.log_artifact(os.path.join(args.output, "model.pkl"))

if __name__ == "__main__":
    main()