import argparse

import pandas as pd
import tabulate

from continualnat.metrics import build_acc_matrix, compute_acc, compute_bwt, compute_fwt


def parse_arguments(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Path to where the BLEU scores are saved")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt_parser = parse_arguments()
    model_name = opt_parser.m
    exp0 = pd.read_csv(f"{model_name}_exp0_wmt.csv")
    exp1 = pd.read_csv(f"{model_name}_exp1_wmt.csv")
    exp2 = pd.read_csv(f"{model_name}_exp2_wmt.csv")
    translation_directions = ["en->de", "de->en", "en->fr", "fr->en", "en->es", "es->en"]
    exps = [exp0, exp1, exp2]
    bleu_scores_exps = {}
    for i, exp in enumerate(exps):
        bleu_scores_exp = exp[translation_directions]
        bleu_scores_exp = bleu_scores_exp.iloc[0].to_list()
        bleu_scores_exp = [bleu_scores_exp[:2], bleu_scores_exp[2:4], bleu_scores_exp[4:]]
        bleu_scores_exps[i] = bleu_scores_exp

    acc_matrix = build_acc_matrix(bleu_scores_exps)
    bleu_scores_tabulated = tabulate.tabulate(acc_matrix)
    print(f"Average BLEU scores:\n{bleu_scores_tabulated}\n")
    acc = compute_acc(bleu_scores_exps)
    bwt = compute_bwt(bleu_scores_exps)
    fwt = compute_fwt(bleu_scores_exps)
    print(f"ACC: {acc}\nBWT: {bwt}\nFWT: {fwt}")
