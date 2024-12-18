import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def split(df_path: str, train_path: str, test_path: str, seed: int):
    # Load the data
    data = pd.read_csv(df_path)
    # Split the dataset into train and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Preprocessing",
        description="Stuff to do with the dataset",
    )

    parser.add_argument(
        "--input", "-i", type=str, default="catalogue", help="Input folder"
    )
    parser.add_argument(
        "--train-path", type=str, default="train.csv", help="train file", dest="train"
    )
    parser.add_argument("--seed", type=int, default=1337, help="Seed value")
    parser.add_argument(
        "--test-path", type=str, default="test.csv", help="test file", dest="test"
    )
    args = parser.parse_args()
    split(args.input, args.train, args.test, args.seed)
