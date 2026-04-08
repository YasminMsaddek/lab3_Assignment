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

def resolve_parquet_path(path: str) -> str:
    if os.path.isdir(path):
        parquet_path = os.path.join(path, "data.parquet")
    else:
        parquet_path = path
    return parquet_path

def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_parquet_path(path))

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = (out["overall"] >= 4).astype(int)
    return out

def expand_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        series = out[col]
        non_null = series.dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], (list, tuple)):
            expanded = pd.DataFrame(series.tolist(), index=out.index)
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            out = out.drop(columns=[col]).join(expanded)
    return out

def build_features(df: pd.DataFrame, reference_columns=None) -> pd.DataFrame:
    out = expand_list_columns(df)
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")
    out = out.select_dtypes(include=["number", "bool"]).astype(float)
    out = out.fillna(0)
    if reference_columns is not None:
        for col in reference_columns:
            if col not in out.columns:
                out[col] = 0.0
        out = out[list(reference_columns)]
    return out

def evaluate(model, X, y, split: str):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    metrics = {
        f"{split}_accuracy": accuracy_score(y, preds),
        f"{split}_auc": roc_auc_score(y, probs),
    }
    mlflow.log_metrics(metrics)

def main():
    args = parse_args()
    with mlflow.start_run():
        mlflow.log_param("C", args.C)
        mlflow.log_param("solver", args.solver)

        # Process Train and clear memory
        train_df = create_labels(load_data(args.train_data))
        X_train = build_features(train_df)
        y_train = train_df["label"]
        feature_columns = X_train.columns.tolist()
        del train_df
        gc.collect() 

        # Process Val and clear memory
        val_df = create_labels(load_data(args.val_data))
        X_val = build_features(val_df, feature_columns)
        y_val = val_df["label"]
        del val_df
        gc.collect()

        # Process Test and clear memory
        test_df = create_labels(load_data(args.test_data))
        X_test = build_features(test_df, feature_columns)
        y_test = test_df["label"]
        del test_df
        gc.collect()

        model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver=args.solver, random_state=42)
        model.fit(X_train, y_train)

        evaluate(model, X_train, y_train, "train")
        evaluate(model, X_val, y_val, "val")
        evaluate(model, X_test, y_test, "test")

        os.makedirs(args.output, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": feature_columns}, os.path.join(args.output, "model.pkl"))
        mlflow.log_artifact(os.path.join(args.output, "model.pkl"))

if __name__ == "__main__":
    main()