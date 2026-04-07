import json
import os
import joblib
import pandas as pd

model = None
feature_columns = None

DROP_COLS = {
    "asin", "reviewerID", "overall", "label",
    "reviewText", "reviewText_x", "reviewText_y",
    "title", "title_x", "title_y",
    "summary", "summary_x", "summary_y",
    "brand", "brand_x", "brand_y"
}


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
    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0.0
    return out[feature_columns]


def init():
    global model, feature_columns
    model_dir = os.getenv("AZUREML_MODEL_DIR", "")
    candidates = [
        os.path.join(model_dir, "model.pkl"),
        os.path.join(model_dir, "model_output", "model.pkl"),
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(f"model.pkl not found under AZUREML_MODEL_DIR={model_dir}")

    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_columns = payload["feature_columns"]
    else:
        model = payload
        feature_columns = []


def run(raw_data):
    try:
        data = json.loads(raw_data)
        records = data.get("data", data)
        df = pd.DataFrame(records)
        X = build_features(df)
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
