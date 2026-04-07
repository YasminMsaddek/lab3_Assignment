import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # Updated ratios
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--deploy_ratio", type=float, default=0.1)

    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    # --------------------------------------------------
    # STEP 1: Create deployment split (latest data)
    # --------------------------------------------------
    if "review_year" in df.columns:
        df = df.sort_values("review_year")

        deploy_size = int(len(df) * args.deploy_ratio)
        deploy_df = df.tail(deploy_size)
        remaining_df = df.iloc[:-deploy_size]
    else:
        # fallback if review_year not available
        remaining_df, deploy_df = train_test_split(
            df,
            test_size=args.deploy_ratio,
            random_state=args.seed,
            shuffle=True
        )

    # --------------------------------------------------
    # STEP 2: Train / Val / Test split (from remaining)
    # --------------------------------------------------
    train_df, temp_df = train_test_split(
        remaining_df,
        test_size=(1 - args.train_ratio),
        random_state=args.seed,
        shuffle=True
    )

    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        random_state=args.seed,
        shuffle=True
    )

    # --------------------------------------------------
    # STEP 3: Save outputs
    # --------------------------------------------------
    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.deploy_out, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"))
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"))
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"))
    deploy_df.to_parquet(os.path.join(args.deploy_out, "data.parquet"))

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))
    print("Deployment rows:", len(deploy_df))


if __name__ == "__main__":
    main()