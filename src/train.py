import argparse
import os
import time
import joblib
import mlflow
import pandas as pd
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

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

def load_data(path: str, sample_frac=1.0) -> pd.DataFrame:
    target = os.path.join(path, "data.parquet") if os.path.isdir(path) else path
    df = pd.read_parquet(target)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    return df

def process_split(path: str, sample_frac=1.0, ref_cols=None):
    """Loads, downsamples, and expands features to save RAM."""
    df = load_data(path, sample_frac)
    df["label"] = (df["overall"] >= 4).astype(int)
    y = df["label"].values
    
    # Expand lists (SBERT vectors)
    for col in list(df.columns):
        series = df[col]
        non_null = series.dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], (list, tuple)):
            expanded = pd.DataFrame(series.tolist(), index=df.index)
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            df = df.drop(columns=[col]).join(expanded)
    
    # Filter to numeric only
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=["number", "bool"]).astype(float).fillna(0)
    
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
    with mlflow.start_run():
        mlflow.log_param("C", args.C)
        mlflow.log_param("solver", args.solver)

        # Downsample train significantly to fit in 14GB RAM
        print("Processing Training data (30% sample)...")
        X_train, y_train, feature_columns = process_split(args.train_data, sample_frac=0.3)

        print("Processing Validation data...")
        X_val, y_val, _ = process_split(args.val_data, sample_frac=1.0, ref_cols=feature_columns)

        print("Processing Test data...")
        X_test, y_test, _ = process_split(args.test_data, sample_frac=1.0, ref_cols=feature_columns)

        print(f"Training Logistic Regression (Rows: {len(X_train)})...")
        model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver=args.solver, random_state=42)
        model.fit(X_train, y_train)

        # Metrics
        val_acc = accuracy_score(y_val, model.predict(X_val))
        mlflow.log_metric("val_accuracy", val_acc)
        print(f"val_accuracy: {val_acc}")

        # Save artifacts
        os.makedirs(args.output, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": feature_columns}, os.path.join(args.output, "model.pkl"))
        mlflow.log_artifact(os.path.join(args.output, "model.pkl"))

if __name__ == "__main__":
    main()