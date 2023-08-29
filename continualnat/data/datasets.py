from typing import Dict, Iterator, Set, Union

import datasets
import numpy as np
import torch
from transformers import MBartTokenizer, MBartTokenizerFast
from torch.utils.data import Dataset, IterableDataset

from continualnat.utils.utils import MBART_LANG_MAP


class TranslationDatasetCore:

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: Union[datasets.Dataset, datasets.IterableDataset],
                 tokenizer: Union[MBartTokenizer, MBartTokenizerFast],
                 max_length: int = 128,
                 use_cls_token: bool = False,
                 skip_idxs: Set[int] = None) -> None:
        """
        Base class for all the translation datasets.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the Huggingface dataset to wrap.
        :param tokenizer: the tokenizer used by the collator when called.
        :param max_length: maximum allowed length for the tokenized sentences (default=128)
        :param use_cls_token: whether to add the cls token at the beginnning of the source sentences (default=False).
        :param skip_idxs: the indices to skip (default=None).
        """
        # Checks before initializing everything
        if "translation" not in dataset.features.keys():
            raise ValueError("You should use a dataset suitable for the translation task.")

        if not hasattr(tokenizer, "src_lang") or not hasattr(tokenizer, "tgt_lang"):
            raise ValueError("You should use a tokenizer that can has \"source_lang\" and \"tgt_lang\" defined.")

        # Source and target languages
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_lang_code = MBART_LANG_MAP[src_lang]
        self.tgt_lang_code = MBART_LANG_MAP[tgt_lang]

        # Dataset
        self.dataset = dataset

        # Tokenizer
        self.tokenizer = tokenizer
        self.tokenizer_state = {"truncation": True, "add_special_tokens": True, "padding": "longest",
                                "max_length": max_length, "return_tensors": "pt"}
        self.max_tokens = max_length
        self.use_cls_token = use_cls_token

        # Dataset's indexes that should be skipped (duplicated, corrupted or unwanted sentences)
        self.skip_idxs = set() if skip_idxs is None else skip_idxs

    def tokenize_pair(self, sentence_pair: Dict[str, str]) -> Dict[str, Union[torch.Tensor, str]]:
        src_sentence = sentence_pair[self.src_lang].strip()
        src_sentence = self.tokenizer.cls_token + src_sentence if self.use_cls_token else src_sentence
        tgt_sentence = sentence_pair[self.tgt_lang].strip()
        self.tokenizer.src_lang = self.src_lang_code
        self.tokenizer.tgt_lang = self.tgt_lang_code
        input_ids = self.tokenizer(src_sentence, **self.tokenizer_state)
        labels = self.tokenizer(text_target=tgt_sentence, **self.tokenizer_state)
        return {"input_ids": input_ids["input_ids"], "labels": labels["input_ids"], "reference": tgt_sentence}


class TranslationDataset(TranslationDatasetCore, Dataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.Dataset,
                 tokenizer: Union[MBartTokenizer, MBartTokenizerFast],
                 max_length: Union[int, None] = None,
                 use_cls_token: bool = False,
                 skip_idxs: Set[int] = None,
                 fill_to_max_length: bool = False) -> None:
        """
        Translation dataset defined by source and target languages.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the Huggingface dataset to wrap.
        :param tokenizer: the tokenizer used by the collator when called.
        :param max_length: maximum allowed length fot the tokenized sentences (default=None)
        :param use_cls_token: whether to add the cls token at the beginnning of the source sentences (default=False).
        :param skip_idxs: the indices to skip (default=None).
        """
        TranslationDatasetCore.__init__(self, src_lang, tgt_lang, dataset, tokenizer, max_length, use_cls_token,
                                        skip_idxs)
        Dataset.__init__(self)
        self.fill_to_max_length = fill_to_max_length
        if fill_to_max_length:
            self.tokenizer_state["add_special_tokens"] = False
            self.tokenizer_state["max_length"] -= 2  # just a workaround

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        while idx in self.skip_idxs:
            idx = np.random.randint(0, self.__len__())

        sentence_pair = self.dataset[idx]["translation"]

        # Tokenized sentence pair
        tokenized_sentence_pair = self.tokenize_pair(sentence_pair)
        input_ids: torch.Tensor = tokenized_sentence_pair["input_ids"]
        labels: torch.Tensor = tokenized_sentence_pair["labels"]
        reference = tokenized_sentence_pair["reference"]

        # Fill the sentence until we reach a determined number of tokens
        if self.fill_to_max_length:
            src_seq_len = input_ids.size(-1)
            tgt_seq_len = labels.size(-1)
            sep_token_id = torch.tensor([self.tokenizer.sep_token_id]).unsqueeze(0)
            while src_seq_len < self.max_tokens * 0.8 and tgt_seq_len < self.max_tokens * 0.8:
                concat_idx = np.random.randint(0, self.__len__())
                sentence_pair_to_concat = self.dataset[concat_idx]["translation"]
                tokenized_sentence_pair_to_concat = self.tokenize_pair(sentence_pair_to_concat)
                input_ids_to_concat = tokenized_sentence_pair_to_concat["input_ids"]
                labels_to_concat = tokenized_sentence_pair_to_concat["labels"]
                concat_src_seq_len = src_seq_len + input_ids_to_concat.size(-1)
                concat_tgt_seq_len = tgt_seq_len + labels_to_concat.size(-1)
                if concat_src_seq_len < self.max_tokens - 2 and concat_tgt_seq_len < self.max_tokens - 2:
                    # Concat the previous tokens with the sep token and the new ones
                    input_ids = torch.cat([input_ids, sep_token_id, input_ids_to_concat], dim=-1)
                    labels = torch.cat([labels, sep_token_id, labels_to_concat], dim=-1)
                    reference = reference + " " + tokenized_sentence_pair_to_concat["reference"]
                    src_seq_len = concat_src_seq_len
                    tgt_seq_len = concat_tgt_seq_len

            # Concat the input ids and labels to their respective language token
            src_lang_token = torch.tensor([self.tokenizer.eos_token_id,
                                           self.tokenizer.lang_code_to_id[self.src_lang_code]]).unsqueeze(0)
            tgt_lang_token = torch.tensor([self.tokenizer.eos_token_id,
                                           self.tokenizer.lang_code_to_id[self.tgt_lang_code]]).unsqueeze(0)
            input_ids = torch.cat([input_ids, src_lang_token], dim=-1)
            labels = torch.cat([labels, tgt_lang_token], dim=-1)

        # Compute the special tokens mask for the input ids and the labels
        input_ids_special_mask = self.tokenizer.get_special_tokens_mask(input_ids[0], already_has_special_tokens=True)
        input_ids_special_mask = torch.tensor(input_ids_special_mask).unsqueeze(0)
        labels_special_mask = self.tokenizer.get_special_tokens_mask(labels[0], already_has_special_tokens=True)
        labels_special_mask = torch.tensor(labels_special_mask).unsqueeze(0)
        return {"input_ids": input_ids, "labels": labels, "reference": reference,
                "input_ids_special_mask": input_ids_special_mask, "labels_special_mask": labels_special_mask}


class IterableTranslationDataset(TranslationDatasetCore, IterableDataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.IterableDataset,
                 tokenizer: Union[MBartTokenizer, MBartTokenizerFast],
                 max_length: Union[int, None] = None,
                 use_cls_token: bool = False,
                 skip_idxs: Set[int] = None) -> None:
        """
        Translation dataset for iterable Hugginface datasets, defined by source and target languages.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the iterable Hugginface dataset to wrap.
        :param tokenizer: the tokenizer used by the collator when called.
        :param max_length: maximum allowed length fot the tokenized sentences (default=None).
        :param use_cls_token: whether to add the cls token at the beginnning of the source sentences (default=False).
        :param skip_idxs: the indices to skip (default=None).
        """
        TranslationDatasetCore.__init__(self, src_lang, tgt_lang, dataset, tokenizer, max_length, use_cls_token,
                                        skip_idxs)
        IterableDataset.__init__(self)

    def __iter__(self) -> Iterator:
        for sentence_pair_langs in self.dataset:
            if "id" in sentence_pair_langs.keys():
                sentence_pair_id = sentence_pair_langs["id"]
                if sentence_pair_id in self.skip_idxs:
                    continue

            sentence_pair = sentence_pair_langs["translation"]
            yield self.tokenize_pair(sentence_pair)
