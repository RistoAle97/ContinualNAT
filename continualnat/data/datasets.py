import warnings

import datasets
import numpy as np
import torch
from transformers import MBartTokenizer, MBartTokenizerFast
from torch.utils.data import Dataset, Subset

from continualnat.utils.utils import MBART_LANG_MAP, NLLB_FLORES200_LANG_MAP


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        dataset: datasets.Dataset | Subset,
        tokenizer: MBartTokenizer | MBartTokenizerFast,
        max_length: int | None = None,
        use_nllb_lang_map: bool = False,
        use_cls_token: bool = False,
        skip_idxs: set[int] = None,
        fill_to_max_length: bool = False,
        lang_tokens_only_encoder: bool = False,
    ) -> None:
        """
        Translation dataset defined by source and target languages.
        :param src_lang: the source language in ISO 639-1 format (e.g., de for german).
        :param tgt_lang: the target language in ISO 639-1 format (e.g., de for german).
        :param dataset: the Huggingface dataset to wrap.
        :param tokenizer: the tokenizer used by the collator when called.
        :param max_length: maximum allowed length fot the tokenized sentences (default=None).
        :param use_nllb_lang_map: whether to use the nllb lang map (e.g., de -> deu_Latn) for extracting the source and
            target sentences from the dataset (default=False).
        :param use_cls_token: whether to add the cls token at the beginnning of the source sentences (default=False).
        :param skip_idxs: the indices to skip (default=None).
        :param fill_to_max_length: whether to fill a tensor with multiple sentences until the max_length is reached
            (default=False).
        :param lang_tokens_only_encoder: whether to use the language tokens only for the input ids, keep in mind that
            doing so the target language tokens will succeed the source one inside the input ids (default=False).
        """
        Dataset.__init__(self)

        # Checks before initializing everything
        if not hasattr(tokenizer, "src_lang") or not hasattr(tokenizer, "tgt_lang"):
            raise ValueError("You should use a tokenizer that has \"source_lang\" and \"tgt_lang\" defined.")

        # Source and target languages
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_nllb_lang_map = use_nllb_lang_map
        if self.use_nllb_lang_map:
            self.src_lang = NLLB_FLORES200_LANG_MAP[src_lang]
            self.tgt_lang = NLLB_FLORES200_LANG_MAP[tgt_lang]

        self.src_lang_code = MBART_LANG_MAP[src_lang]
        self.tgt_lang_code = MBART_LANG_MAP[tgt_lang]

        # Dataset
        self.dataset = dataset

        # Tokenizer
        self.tokenizer = tokenizer
        self.tokenizer_state_src = {
            "truncation": True,
            "add_special_tokens": True,
            "padding": "longest",
            "max_length": max_length,
            "return_tensors": "pt",
        }
        self.tokenizer_state_tgt = self.tokenizer_state_src.copy()
        self.max_tokens = max_length
        self.use_cls_token = use_cls_token
        self.max_length = max_length

        # Dataset's indexes that should be skipped (duplicated, corrupted or unwanted sentences)
        self.skip_idxs = set() if skip_idxs is None else skip_idxs
        if "id" not in dataset.features.keys() and skip_idxs:
            warnings.warn("You have passed some indices to skip but the dataset to wrap has no such feature.")

        # Update the tokenizer states if filling to max length
        self.fill_to_max_length = fill_to_max_length
        if fill_to_max_length:
            self.tokenizer_state_src["add_special_tokens"] = False
            self.tokenizer_state_tgt["add_special_tokens"] = False

            # Workaround for keeping the defined max length
            self.tokenizer_state_src["max_length"] -= 2
            self.tokenizer_state_tgt["max_length"] -= 2

        # Update the tokenizer states if using the language tokens only for the input ids
        self.lang_tokens_only_encoder = lang_tokens_only_encoder
        if lang_tokens_only_encoder:
            self.tokenizer_state_src["add_special_tokens"] = False
            self.tokenizer_state_tgt["add_special_tokens"] = False

            # Workaround for keeping the defined max length
            self.tokenizer_state_src["max_length"] -= 3
            self.tokenizer_state_tgt["max_length"] -= 1

    def __len__(self) -> int:
        return len(self.dataset)

    def __tokenize_pair(self, sentence_pair: dict[str, str]) -> dict[str, torch.Tensor | str]:
        if self.use_nllb_lang_map:
            src_lang_feature = f"sentence_{self.src_lang}"
            tgt_lang_feature = f"sentence_{self.tgt_lang}"
        else:
            src_lang_feature = self.src_lang
            tgt_lang_feature = self.tgt_lang

        # Split the sentence pair into source and target sentences
        src_sentence = sentence_pair[src_lang_feature].strip()
        src_sentence = self.tokenizer.cls_token + src_sentence if self.use_cls_token else src_sentence
        tgt_sentence = sentence_pair[tgt_lang_feature].strip()

        # Define the source and target language for the tokenizer
        self.tokenizer.src_lang = self.src_lang_code
        self.tokenizer.tgt_lang = self.tgt_lang_code

        # Tokenize the source and target sentences
        input_ids = self.tokenizer(src_sentence, **self.tokenizer_state_src)
        labels = self.tokenizer(text_target=tgt_sentence, **self.tokenizer_state_tgt)

        return {"input_ids": input_ids["input_ids"], "labels": labels["input_ids"], "reference": tgt_sentence}

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        while idx in self.skip_idxs:
            idx = np.random.randint(0, self.__len__())

        sentence_pair = self.dataset[idx]
        if "translation" in sentence_pair:
            sentence_pair = sentence_pair["translation"]

        # Tokenized sentence pair
        tokenized_sentence_pair = self.__tokenize_pair(sentence_pair)
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
                tokenized_sentence_pair_to_concat = self.__tokenize_pair(sentence_pair_to_concat)
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
            src_lang_token = torch.tensor(
                [self.tokenizer.eos_token_id, self.tokenizer.lang_code_to_id[self.src_lang_code]]
            ).unsqueeze(0)
            tgt_lang_token = torch.tensor(
                [self.tokenizer.eos_token_id, self.tokenizer.lang_code_to_id[self.tgt_lang_code]]
            ).unsqueeze(0)
            input_ids = torch.cat([input_ids, src_lang_token], dim=-1)
            labels = torch.cat([labels, tgt_lang_token], dim=-1)

        if self.lang_tokens_only_encoder:
            src_lang_token = torch.tensor(
                [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.lang_code_to_id[self.src_lang_code],
                    self.tokenizer.lang_code_to_id[self.tgt_lang_code],
                ]
            ).unsqueeze(0)
            tgt_lang_token = torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)
            input_ids = torch.cat([input_ids, src_lang_token], dim=-1)
            labels = torch.cat([labels, tgt_lang_token], dim=-1)

        # Compute the special tokens mask for the input ids and the labels
        input_ids_special_mask = self.tokenizer.get_special_tokens_mask(input_ids[0], already_has_special_tokens=True)
        input_ids_special_mask = torch.tensor(input_ids_special_mask).unsqueeze(0)
        labels_special_mask = self.tokenizer.get_special_tokens_mask(labels[0], already_has_special_tokens=True)
        labels_special_mask = torch.tensor(labels_special_mask).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "reference": reference,
            "input_ids_special_mask": input_ids_special_mask,
            "labels_special_mask": labels_special_mask,
        }
