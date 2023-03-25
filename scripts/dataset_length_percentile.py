import torch
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from typing import Dict


def build_cumsum_bins(bins: Dict[int, int]):
    cumsum_bins = torch.zeros(max(bins.keys()) + 1).int()
    for bin_idx in list(bins.keys()):
        cumsum_bins[bin_idx] = bins[bin_idx]

    cumsum_bins = torch.cumsum(cumsum_bins, dim=0)
    return cumsum_bins


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source language code")
    parser.add_argument("--tgt", type=str, help="target language code")
    parser.add_argument("--p", nargs="+", type=float, default=[0.9, 0.95, 0.98, 0.99],
                        help="the length percentiles")
    parser.add_argument("--path", type=str, default="", help="where to save the csv file")
    args = parser.parse_args()
    src_lang = args.src
    tgt_lang = args.tgt
    assert src_lang is not None and tgt_lang is not None
    percentiles = torch.tensor(args.p)
    path = args.path
    if percentiles.any() > 1 or percentiles.any() < 0:
        raise ValueError("Percentiles must be floats in [0, 1].")

    # Load dataset
    dataset = load_dataset("yhavinga/ccmatrix", f"{src_lang}-{tgt_lang}", split="train",
                           cache_dir="/disk1/a.ristori/datasets/ccmatrix", verification_mode="no_checks")

    # Build dict of bins
    src_bins: Dict[int, int] = {}
    tgt_bins: Dict[int, int] = {}
    for sentence_pair in tqdm(dataset):
        src_sentence = sentence_pair["translation"][src_lang]
        tgt_sentence = sentence_pair["translation"][tgt_lang]
        src_length = len(src_sentence.split())
        tgt_length = len(tgt_sentence.split())
        if src_length not in src_bins.keys():
            src_bins.update({src_length: 1})
        else:
            src_bins[src_length] += 1

        if tgt_length not in tgt_bins.keys():
            tgt_bins.update({tgt_length: 1})
        else:
            tgt_bins[tgt_length] += 1

    # Compute cumulative sums
    src_bins_cumsum = build_cumsum_bins(src_bins)
    tgt_bins_cumsum = build_cumsum_bins(tgt_bins)

    # Build the percentiles dataframe
    df_percentiles = {"percentiles": percentiles.tolist(), f"{src_lang}_length": [], f"{tgt_lang}_length": []}
    for percentile in percentiles:
        p: int = percentile.item()
        src_percentiles = src_bins_cumsum / src_bins_cumsum[-1]
        tgt_percentiles = tgt_bins_cumsum / tgt_bins_cumsum[-1]
        src_length = (src_percentiles > p).nonzero()[0]
        tgt_length = (tgt_percentiles > p).nonzero()[0]
        df_percentiles[f"{src_lang}_length"].append(src_length.item())
        df_percentiles[f"{tgt_lang}_length"].append(tgt_length.item())

    df_percentiles = pd.DataFrame.from_dict(df_percentiles)

    # Save the dataframe as a csv
    df_percentiles.to_csv(f"{path}/percentiles_{src_lang}_{tgt_lang}.csv")
    print(f"Check the results inside {path}/percentiles_{src_lang}_{tgt_lang}.csv")
