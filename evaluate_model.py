import argparse

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import MBartTokenizerFast

from continualnat.data import TranslationDataset
from continualnat.metrics import *
from continualnat.models import *
from continualnat.utils import NLLB_FLORES200_LANG_MAP


def parse_arguments(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Where the model state dict is saved")
    parser.add_argument("-lp", nargs="+", default=["en-de", "en-fr", "en-es"], type=str, help="Lang pairs to consider")
    parser.add_argument("-bsz", default=32, type=int, help="The batch size used during decoding")
    parser.add_argument("-s", action="store_true", help="Whether to save the scores in a csv file")
    parser.add_argument("-v", action="store_true", help="Whether to print the BLEU scores")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    # Set-up
    torch.set_float32_matmul_precision("medium")

    # Parse command line arguments
    opt_parser = parse_arguments()
    model_to_load: str = opt_parser.m
    lang_pairs = set(opt_parser.lp)
    bsz: int = opt_parser.bsz
    save_scores: bool = opt_parser.s
    verbose: bool = opt_parser.v

    # Define the unique language pairs
    unique_lang_pairs = []
    unique_lang_pairs_flores200 = []
    available_lang_pairs = {"en-de", "de-en", "en-fr", "fr-en", "en-es", "es-en"}
    for lang_pair in lang_pairs:
        if lang_pair not in available_lang_pairs:
            raise ValueError(f"{lang_pair} is not a valid language pair. The pairs availble are {available_lang_pairs}")

        first_lang, second_lang = lang_pair.split("-")
        first_lang_flores200 = NLLB_FLORES200_LANG_MAP[first_lang]
        second_lang_flores200 = NLLB_FLORES200_LANG_MAP[second_lang]
        if first_lang != "en":
            unique_lang_pair = f"{first_lang}-{second_lang}"
            unique_lang_pair_flores200 = f"{second_lang_flores200}-{first_lang_flores200}"
        else:
            unique_lang_pair = f"{second_lang}-{first_lang}"
            unique_lang_pair_flores200 = f"{first_lang_flores200}-{second_lang_flores200}"

        unique_lang_pairs.append(unique_lang_pair)
        unique_lang_pairs_flores200.append(unique_lang_pair_flores200)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tokenizer and some useful tokens
    tokenizer = MBartTokenizerFast(
        tokenizer_file="tokenizers/sp_32k.json", model_max_length=1024,
        cls_token="<length>"
    )
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id

    # Load the model
    model_state_dict = torch.load(model_to_load)
    model_config = TransformerConfig(
        vocab_size=len(tokenizer),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    model = Transformer(model_config)
    model.load_state_dict(model_state_dict)
    model.to(device)

    # Some parameters for the translation datasets
    lang_tokens_only_encoder = isinstance(model, GLAT)
    use_cls_token = hasattr(model, "length_token_id") and model.length_token_id is not None

    # Load the datasets
    test_datasets = []
    for lang_pair, lang_pair_flores200 in zip(unique_lang_pairs, unique_lang_pairs_flores200):
        first_lang, second_lang = lang_pair.split("-")
        wmt_dataset_hf = load_dataset(
            path="thesistranslation/wmt14",
            name=lang_pair,
            cache_dir="/disk1/a.ristori/datasets/wmt14",
            verification_mode="no_checks",
        )
        flores200_dataset_hf = load_dataset(
            path="facebook/flores",
            name=lang_pair_flores200,
            cache_dir="/disk1/a.ristori/datasets/flores200",
            verification_mode="no_checks",
        )
        shared_parameters = {
            "tokenizer": tokenizer,
            "max_length": 128,
            "use_cls_token": use_cls_token,
            "lang_tokens_only_encoder": lang_tokens_only_encoder,
        }
        wmt_test_first_second = TranslationDataset(
            src_lang=first_lang, tgt_lang=second_lang, dataset=wmt_dataset_hf["test"], **shared_parameters
        )
        wmt_test_second_first = TranslationDataset(
            src_lang=second_lang, tgt_lang=first_lang, dataset=wmt_dataset_hf["test"], **shared_parameters
        )
        flores200_devtest_first_second = TranslationDataset(
            src_lang=first_lang,
            tgt_lang=second_lang,
            dataset=flores200_dataset_hf["devtest"],
            use_nllb_lang_map=True,
            **shared_parameters,
        )
        flores200_devtest_second_first = TranslationDataset(
            src_lang=second_lang,
            tgt_lang=first_lang,
            dataset=flores200_dataset_hf["devtest"],
            use_nllb_lang_map=True,
            **shared_parameters,
        )
        test_datasets.append({"wmt": wmt_test_second_first, "flores200": flores200_devtest_second_first})
        test_datasets.append({"wmt": wmt_test_first_second, "flores200": flores200_devtest_first_second})

    # Compute the BLEU scores
    generation_parameters = {
        "tokenizer": tokenizer,
        "bsz": bsz,
        "metric_tokenize": {"13a", "intl"},
    }
    metric_tokenize = {"13a", "intl"}
    bleu_scores_wmt_df = {"tokenizer": ["13a", "intl"]}
    bleu_scores_flores200_df = {"tokenizer": ["13a", "intl"]}
    for dataset_pair in test_datasets:
        src_lang = dataset_pair["wmt"].src_lang
        tgt_lang = dataset_pair["wmt"].tgt_lang
        bleu_scores_wmt = compute_sacrebleu(model, dataset_pair["wmt"], **generation_parameters)
        bleu_score_flores200 = compute_sacrebleu(model, dataset_pair["flores200"], **generation_parameters)
        wmt_13a = np.round(bleu_scores_wmt["13a"], 2)
        wmt_intl = np.round(bleu_scores_wmt["intl"], 2)
        flores200_13a = np.round(bleu_score_flores200["13a"], 2)
        flores200_intl = np.round(bleu_score_flores200["intl"], 2)
        if verbose:
            print(
                f"BLEU scores on the WMT14 {src_lang}-{tgt_lang} test\n"
                f"13a: {wmt_13a}\n"
                f"intl: {wmt_intl}\n"
            )
            print(
                f"BLEU scores on the Flores200 {src_lang}-{tgt_lang} devtest\n"
                f"13a: {flores200_13a}\n"
                f"intl: {flores200_intl}\n"
            )

        bleu_scores_wmt_df[f"{src_lang}->{tgt_lang}"] = [wmt_13a, wmt_intl]
        bleu_scores_flores200_df[f"{src_lang}->{tgt_lang}"] = [flores200_13a, flores200_intl]

    if save_scores:
        bleu_scores_wmt_df = pd.DataFrame(bleu_scores_wmt_df)
        bleu_scores_flores200_df = pd.DataFrame(bleu_scores_flores200_df)
        model_version = model_to_load.split("/")[-1]
        bleu_scores_wmt_df.to_csv(f"{model_version}_wmt.csv", index=False)
        bleu_scores_flores200_df.to_csv(f"{model_version}_flores200.csv", index=False)
