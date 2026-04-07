import argparse
import json
import os

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DROP_COLS = {
    "asin", "reviewerID", "overall", "label",
    "reviewText", "reviewText_x", "reviewText_y",
    "title", "title_x", "title_y",
    "summary", "summary_x", "summary_y",
    "brand", "brand_x", "brand_y"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_data", type=str, required=True)
    parser.add_argument("--endpoint_url", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    return parser.parse_args()


def resolve_parquet_path(path: str) -> str:
    return os.path.join(path, "data.parquet") if os.path.isdir(path) else path


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = (out["overall"] >= 4).astype(int)
    return out


def expand_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        non_null = out[col].dropna()
        if non_null.empty:
            continue
        sample = non_null.iloc[0]
        if isinstance(sample, (list, tuple)):
            expanded = pd.DataFrame(out[col].tolist(), index=out.index)
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            out = out.drop(columns=[col]).join(expanded)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = expand_list_columns(df)
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")
    out = out.select_dtypes(include=["number", "bool"]).astype(float)
    out = out.fillna(0)
    return out


def main():
    args = parse_args()
    df = pd.read_parquet(resolve_parquet_path(args.deploy_data))
    df = create_labels(df)
    X = build_features(df)
    y_true = df["label"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    payload = {"data": X.to_dict(orient="records")}

    response = requests.post(args.endpoint_url, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    result = response.json()

    if "predictions" not in result:
        raise RuntimeError(f"Unexpected endpoint response: {result}")

    preds = result["predictions"]
    print("Deployment accuracy:", accuracy_score(y_true, preds))
    print("Deployment precision:", precision_score(y_true, preds, zero_division=0))
    print("Deployment recall:", recall_score(y_true, preds, zero_division=0))
    print("Deployment f1:", f1_score(y_true, preds, zero_division=0))
    print("Predictions (first 10):", preds[:10])


if __name__ == "__main__":
    main()
