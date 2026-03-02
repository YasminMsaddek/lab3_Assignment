import argparse, os, pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    # Split 70% train, 30% temp
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    # Split temp 50/50 into val and test (15% each of total)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    train.to_parquet(os.path.join(args.train_out, "data.parquet"))
    val.to_parquet(os.path.join(args.val_out, "data.parquet"))
    test.to_parquet(os.path.join(args.test_out, "data.parquet"))

if __name__ == "__main__":
    main()