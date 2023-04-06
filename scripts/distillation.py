import torch
import argparse
import evaluate
import pandas as pd
import os
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="en", help="source language")
    parser.add_argument("--tgt", type=str, default="de", help="target language")
    parser.add_argument("--dataset", type=str, default="yhavinga/ccmatrix", help="huggingface dataset to distill")
    parser.add_argument("--size", type=int, default=100000, help="number of sentences to consider from the dataset")
    parser.add_argument("--cachedir", type=str, default="D:/MasterDegreeThesis/datasets/ccmatrix",
                        help="dataset's cache directory")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--evaluate", action="store_true", help="whether to evaluate the teacher's translations")
    return parser.parse_args()


if __name__ == '__main__':
    opt_parser = parse_arguments()
    src_lang = opt_parser.src
    tgt_lang = opt_parser.tgt
    lang_pair = src_lang + "-" + tgt_lang
    dataset = opt_parser.dataset
    dataset_size = opt_parser.size
    cache_dir = opt_parser.cachedir
    batch_size = opt_parser.batch
    evaluate_teacher = opt_parser.evaluate

    # Set-up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    opus_mt_model: MarianMTModel = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_pair}").to(device)
    opus_mt_tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_pair}")

    # Load dataset
    dataset_to_distill = load_dataset(dataset, f"{lang_pair}", cache_dir=f"{cache_dir}_{src_lang}_{tgt_lang}",
                                      split=f"train[:{dataset_size}]", verification_mode="no_checks")

    # Extract target sentences
    tgt_sentences = [tgt_sentence[tgt_lang] for tgt_sentence in dataset_to_distill["translation"]]

    # Compute predictions
    dataloader_opus_mt = DataLoader(dataset_to_distill["translation"], batch_size=batch_size)
    opus_mt_predictions = []
    for batch in tqdm(dataloader_opus_mt, "Tokens prediction"):
        src_batch = batch[src_lang]
        batch_tokens = opus_mt_tokenizer(src_batch, padding=True, return_tensors="pt").to(device)
        output = opus_mt_model.generate(**batch_tokens, max_new_tokens=opus_mt_model.config.max_length)
        opus_mt_predictions.append(output)

    # Detokenize predictions
    translations = []
    for prediction_batch in tqdm(opus_mt_predictions, "Tokens detokenization"):
        translation = opus_mt_tokenizer.batch_decode(prediction_batch, skip_special_tokens=True)
        translations.append(translation)

    translations = [translation for batch in translations for translation in batch]

    # Evaluate teacher's translations and save scores inside a csv file
    if evaluate_teacher:
        bleu_metric = evaluate.load("bleu")
        chrf_metric = evaluate.load("chrf")

        # Compute scores
        bleu_score = bleu_metric.compute(predictions=translations, references=tgt_sentences)["bleu"] * 100
        chrf_score = chrf_metric.compute(predictions=translations, references=tgt_sentences)["score"]
        df_scores = {
            "teacher_model": "Helsinki-NLP/opus-mt-{0}".format(lang_pair),
            "lang_pair": lang_pair,
            "bleu": [bleu_score],
            "chrf": [chrf_score]
        }

        # Save scores
        df_scores = pd.DataFrame(df_scores)
        if os.path.exists("../data/distillation_teacher_scores.csv"):
            df_teacher_scores: pd.DataFrame = pd.read_csv("../data/distillation_teacher_scores.csv", index_col=0)
            df_teacher_scores = pd.concat([df_teacher_scores, df_scores], ignore_index=True).drop_duplicates()
            df_teacher_scores.to_csv("../data/distillation_teacher_scores.csv")
        else:
            df_scores.to_csv("distillation_teacher_scores.csv")

    # Save translations
    with open(f"data/distilled_dataset_{src_lang}_{tgt_lang}.txt", "w", encoding="utf_8") as datafile:
        for translation in translations:
            datafile.write(translation + "\n")
