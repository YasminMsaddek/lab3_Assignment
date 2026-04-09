import argparse
import os
import joblib
import mlflow
import pandas as pd
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Columns to drop to save memory
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
    target = os.path.join(path, "data.parquet") if os.path.isdir(path) else path
    df = pd.read_parquet(target)
    
    # 1. Downsample to fit in 7GB-14GB RAM
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    
    # 2. Create Target
    df["label"] = (df["overall"] >= 4).astype(int)
    y = df["label"].values
    
    # 3. Efficient List Expansion (SBERT/TF-IDF)
    for col in list(df.columns):
        if not df[col].empty and isinstance(df[col].iloc[0], (list, tuple)):
            # Expand to new dataframe
            expanded = pd.DataFrame(df[col].tolist(), index=df.index)
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            # Drop original list immediately to free RAM
            df = df.drop(columns=[col]).join(expanded)
            del expanded
            gc.collect() 
    
    # 4. Clean up non-numeric data
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=["number", "bool"]).astype(float).fillna(0)
    
    # 5. Feature Alignment (Crucial for Inference)
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
        
        # Use 20% - 30% sample to prevent SIGKILL in DevOps Pipeline
        print("Loading Training data...")
        X_train, y_train, feature_columns = process_split(args.train_data, sample_frac=0.3)

        print("Loading Validation data...")
        X_val, y_val, _ = process_split(args.val_data, ref_cols=feature_columns)

        print(f"Training Model with {len(feature_columns)} features...")
        model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver=args.solver)
        model.fit(X_train, y_train)

        # Logging Metrics for Step VI
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        mlflow.log_metric("val_accuracy", acc)
        print(f"Validation Accuracy: {acc}")
        
        # Save output for Step VII
        os.makedirs(args.output, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": feature_columns}, os.path.join(args.output, "model.pkl"))

if __name__ == "__main__":
    main()