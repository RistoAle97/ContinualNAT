import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, MarianTokenizer

from continualnat.data import distill_dataset, push_distilled_dataset_to_hub


def parse_arguments(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="The source language")
    parser.add_argument("--tgt", type=str, help="The target language")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    # Parse command line arguments
    opt_parser = parse_arguments()
    src_lang = opt_parser.src
    tgt_lang = opt_parser.tgt

    # Define device, directory where the datasets are and will be stored and the repository id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets_dir = "/disk1/a.ristori/datasets/"
    repo_id = f"thesistranslation/distilled-ccmatrix-{src_lang}-{tgt_lang}"
    lang_pair = f"{src_lang}-{tgt_lang}" if tgt_lang == "en" else f"{tgt_lang}-{src_lang}"

    # Load the CCMatrix dataset from the Huggingface hub
    ccmatrix_to_distill = load_dataset(
        path="yhavinga/ccmatrix",
        name=lang_pair,
        split="train[:30000000]",
        cache_dir=f"{datasets_dir}ccmatrix",
        verification_mode="no_checks",
    )

    # Load the Marian-MT tokenizer
    marian_tokenizer: MarianTokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}")

    # Distill the dataset and push it to the Huggingface hub
    distill_dataset(
        teacher=f"ct2-opus-mt-{src_lang}-{tgt_lang}",
        tokenizer=marian_tokenizer,
        dataset=ccmatrix_to_distill,
        dataset_name="ccmatrix",
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device,
        beam_size=4,
        bsz=4096,
        save_dir=f"{datasets_dir}distillation",
    )
    push_distilled_dataset_to_hub(
        cache_dir=f"{datasets_dir}distilled_ccmatrix_to_push",
        repo_id=repo_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        path_name=f"{datasets_dir}distillation/distilled_ccmatrix.{src_lang}_{tgt_lang}",
    )

    # Load the previously pushed dataset in order to test it
    distilled_ccmatrix = load_dataset(
        repo_id, cache_dir=f"{datasets_dir}distilled_ccmatrix", verification_mode="no_checks"
    )
