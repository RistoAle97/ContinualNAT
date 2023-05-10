import torch
import datasets
import numpy as np
from transformers import PreTrainedTokenizerBase, MBartTokenizer, MBartTokenizerFast
from transformers.utils import PaddingStrategy, TensorType
from torch.utils.data import Dataset, IterableDataset
from tqdm.auto import tqdm
from typing import Dict, Iterator, Set, Union
from src.utils import SUPPORTED_LANGS, MBART_LANG_MAP


class TranslationDatasetCore:

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: Union[datasets.Dataset, datasets.IterableDataset],
                 tokenizer: PreTrainedTokenizerBase,
                 truncation: bool = True,
                 max_length: Union[int, None] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 add_special_tokens: bool = True,
                 return_tensors: Union[str, TensorType, None] = "pt",
                 use_cls_token: bool = False,
                 skip_idxs: Set[int] = None) -> None:
        """
        Base class for all the translation datasets.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the Huggingface dataset to wrap.
        :param tokenizer: the tokenizer used by the collator when called.
        :param truncation: whether to apply truncation during the tokenization (default=True).
        :param max_length: maximum allowed length fot the tokenized sentences (default=None)
        :param padding: the padding strategy to apply during the tokenization (defualt=True).
        :param add_special_tokens: whether to use special tokens during the tokenization (default=True).
        :param return_tensors: type of tensors from the tokenizer (default="pt").
        :param use_cls_token: whether to add the cls token at the beginnning of the source sentences (default=False).
        :param skip_idxs: the indices to skip (default=None).
        """
        # Checks before initializing everything
        if "translation" not in dataset.features.keys():
            raise ValueError("You should use a dataset suitable for the translation task.")

        if not hasattr(tokenizer, "src_lang") or not hasattr(tokenizer, "tgt_lang"):
            raise ValueError("You should use a tokenizer that can has \"source_lang\" and \"tgt_lang\" defined.")

        if src_lang not in SUPPORTED_LANGS or tgt_lang not in SUPPORTED_LANGS:
            raise ValueError("There should not be an unsupported language as the source or target language.")

        # Source and target languages
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer_src_lang_code = src_lang
        self.tokenizer_tgt_lang_code = tgt_lang
        if isinstance(tokenizer, MBartTokenizer) or isinstance(tokenizer, MBartTokenizerFast):
            self.tokenizer_src_lang_code = MBART_LANG_MAP[src_lang]
            self.tokenizer_tgt_lang_code = MBART_LANG_MAP[tgt_lang]

        # Dataset and stats about it
        self.dataset = dataset
        self.avg_length_src = 0
        self.avg_length_tgt = 0
        self.max_length_src = 0
        self.max_length_tgt = 0

        # Tokenizer
        self.tokenizer = tokenizer
        self.tokenizer_state = {"truncation": truncation, "max_length": max_length, "padding": padding,
                                "add_special_tokens": add_special_tokens, "return_tensors": return_tensors}
        self.use_cls_token = use_cls_token

        # Dataset's indexes that should be skipped (duplicated, corrupted or unwanted sentences)
        self.skip_idxs = set() if skip_idxs is None else skip_idxs

    def compute_stats(self) -> Dict[str, Union[int, float]]:
        """
        Computes and updates the dataset's stats.
        :return: a dict containing the average and max lengths for both source and target languages.
        """
        for sample in tqdm(self.dataset, "Computing average and max length for source and target"):
            sample: dict
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

    def tokenize_pair(self, sentence_pair: Dict[str, str]) -> Dict[str, torch.Tensor]:
        src_sentence = sentence_pair[self.src_lang]
        src_sentence = self.tokenizer.cls_token + " " + src_sentence if self.use_cls_token else src_sentence
        tgt_sentence = sentence_pair[self.tgt_lang]
        self.tokenizer.src_lang = self.tokenizer_src_lang_code
        self.tokenizer.tgt_lang = self.tokenizer_tgt_lang_code
        input_ids = self.tokenizer(src_sentence, **self.tokenizer_state)
        labels = self.tokenizer(text_target=tgt_sentence, **self.tokenizer_state, return_special_tokens_mask=True)
        return {"input_ids": input_ids["input_ids"], "labels": labels["input_ids"],
                "special_mask_labels": labels["special_tokens_mask"]}


class TranslationDataset(TranslationDatasetCore, Dataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.Dataset,
                 tokenizer: PreTrainedTokenizerBase,
                 truncation: bool = True,
                 max_length: Union[int, None] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 add_special_tokens: bool = True,
                 return_tensors: Union[str, TensorType, None] = "pt",
                 use_cls_token: bool = False,
                 skip_idxs: Set[int] = None) -> None:
        """
        Translation dataset defined by source and target languages.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the Huggingface dataset to wrap.
        :param tokenizer: the tokenizer used by the collator when called.
        :param truncation: whether to apply truncation during the tokenization (default=True).
        :param max_length: maximum allowed length fot the tokenized sentences (default=None)
        :param padding: the padding strategy to apply during the tokenization (defualt=True).
        :param add_special_tokens: whether to use special tokens during the tokenization (default=True).
        :param return_tensors: type of tensors from the tokenizer (default="pt").
        :param use_cls_token: whether to add the cls token at the beginnning of the source sentences (default=False).
        :param skip_idxs: the indices to skip (default=None).
        """
        TranslationDatasetCore.__init__(self, src_lang, tgt_lang, dataset, tokenizer, truncation, max_length, padding,
                                        add_special_tokens, return_tensors, use_cls_token, skip_idxs)
        Dataset.__init__(self)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        while idx in self.skip_idxs:
            idx = np.random.randint(0, self.__len__())

        sentence_pair = self.dataset[idx]["translation"]
        return self.tokenize_pair(sentence_pair)


class IterableTranslationDataset(TranslationDatasetCore, IterableDataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.IterableDataset,
                 tokenizer: PreTrainedTokenizerBase,
                 truncation: bool = True,
                 max_length: Union[int, None] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 add_special_tokens: bool = True,
                 return_tensors: Union[str, TensorType, None] = "pt",
                 use_cls_token: bool = False,
                 skip_idxs: Set[int] = None) -> None:
        """
        Translation dataset for iterable Hugginface datasets, defined by source and target languages.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the iterable Hugginface dataset to wrap.
        :param tokenizer: the tokenizer used by the collator when called.
        :param truncation: whether to apply truncation during the tokenization (default=True).
        :param max_length: maximum allowed length fot the tokenized sentences (default=None)
        :param padding: the padding strategy to apply during the tokenization (defualt=True).
        :param add_special_tokens: whether to use special tokens during the tokenization (default=True).
        :param return_tensors: type of tensors from the tokenizer (default="pt").
        :param use_cls_token: whether to add the cls token at the beginnning of the source sentences (default=False).
        :param skip_idxs: the indices to skip (default=None).
        """
        TranslationDatasetCore.__init__(self, src_lang, tgt_lang, dataset, tokenizer, truncation, max_length, padding,
                                        add_special_tokens, return_tensors, use_cls_token, skip_idxs)
        IterableDataset.__init__(self)

    def __iter__(self) -> Iterator:
        for sentence_pair_langs in self.dataset:
            if "id" in sentence_pair_langs.keys():
                sentence_pair_id = sentence_pair_langs["id"]
                if sentence_pair_id in self.skip_idxs:
                    continue

            sentence_pair = sentence_pair_langs["translation"]
            yield self.tokenize_pair(sentence_pair)


class TextDataset(Dataset):

    def __init__(self, dataset: datasets.Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, str]:
        tgt_sentence = self.dataset[idx]["text"]
        return {"tgt_sentence": tgt_sentence}
