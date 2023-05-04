import argparse
from datasets import load_dataset

if __name__ == "__main__":
    # Parse and retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--src", type=str, help="source language code")
    parser.add_argument("--tgt", type=str, help="target language code")
    parser.add_argument("--path", type=str, help="where to save the dataset")
    args = parser.parse_args()
    dataset_name = args.dataset
    src = args.src
    tgt = args.tgt
    path = args.path

    # Load and save dataset
    dataset = load_dataset(dataset_name, f"{src}-{tgt}", cache_dir=path, verification_mode="no_checks")
    # dataset = load_dataset("cc100", lang=f"{src}", cache_dir="/disk1/a.ristori/datasets/cc100")
