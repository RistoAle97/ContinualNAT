import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, MBartTokenizer
from tqdm import tqdm
from typing import Dict, Union


SUPPORTED_LANGUAGES = {"en": "en_XX", "de": "de_DE", "es": "es_XX", "fr": "fr_XX"}


def translation_collate_fn(batch) -> (torch.Tensor, torch.Tensor):
    src_batch, tgt_batch = [], []
    for sample in batch:
        src_batch.append(sample["input_ids"])
        tgt_batch.append(sample["labels"])

    src_batch = torch.stack(src_batch).squeeze(1)
    tgt_batch = torch.stack(tgt_batch).squeeze(1)
    return src_batch, tgt_batch


class TranslationDataset(Dataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.Dataset,
                 max_length: int = 128,
                 tokenizer: PreTrainedTokenizer = None,
                 use_special_tokens: bool = True) -> None:
        super().__init__()
        if src_lang not in SUPPORTED_LANGUAGES.keys() or tgt_lang not in SUPPORTED_LANGUAGES.keys():
            raise ValueError("One of the chosen languages is not supported.")

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.dataset = dataset
        self.avg_length_src = 0
        self.avg_length_tgt = 0
        self.max_length_src = 0
        self.max_length_tgt = 0
        self.max_length = max_length
        self.special_tokens = use_special_tokens
        self.src_supported_language = SUPPORTED_LANGUAGES[src_lang]
        self.tgt_supported_language = SUPPORTED_LANGUAGES[tgt_lang]
        if tokenizer is not None:
            if tokenizer.src_lang != self.src_supported_language:
                raise ValueError("The source language is not the same defined for the tokenizer.")

            if tokenizer.tgt_lang != self.tgt_supported_language:
                raise ValueError("The target language is not the same defined for the tokenizer.")

            self.tokenizer = tokenizer
        else:
            self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25",
                                                            src_lang=self.src_supported_language,
                                                            tgt_lang=self.tgt_supported_language)

    def compute_stats(self) -> Dict[str, Union[int, float]]:
        for sample in tqdm(self.dataset, "Computing average and max length for source and target"):
            sentences = sample["translation"]
            src_sentence: str = sentences[self.src_lang]
            tgt_sentence: str = sentences[self.tgt_lang]
            length_splitted_src = len(src_sentence.split())
            length_splitted_tgt = len(tgt_sentence.split())
            self.max_length_src = max(self.max_length_src, length_splitted_src)
            self.max_length_tgt = max(self.max_length_tgt, length_splitted_tgt)
            self.avg_length_src += length_splitted_src
            self.avg_length_tgt += length_splitted_tgt

        self.avg_length_src /= len(self.dataset)
        self.avg_length_tgt /= len(self.dataset)
        return {"max_length_src": self.max_length_src, "max_length_tgt": self.max_length_tgt,
                "avg_length_src": self.avg_length_src, "avg_length_tgt": self.avg_length_tgt}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence_pair = self.dataset[idx]["translation"]
        src_sentence = sentence_pair[self.src_lang]
        tgt_sentence = sentence_pair[self.tgt_lang]
        tokenized_sentences = self.tokenizer(src_sentence, text_target=tgt_sentence, truncation=True,
                                             max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {"input_ids": tokenized_sentences["input_ids"], "labels": tokenized_sentences["labels"]}
