import datasets
import torch
import numpy as np
from translation_datasets.TranslationDataset import TranslationDataset
from transformers import MBartTokenizer
from utilities import shift_tokens_right
from typing import Dict


class TranslationDatasetCMLM(TranslationDataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.Dataset,
                 max_length: int = 128,
                 tokenizer: MBartTokenizer = None,
                 use_special_tokens: bool = True,
                 train: bool = False,
                 shift_right_decoder_input: bool = False) -> None:
        super().__init__(src_lang, tgt_lang, dataset, max_length, tokenizer, use_special_tokens)
        self.train = train
        self.shift_right = shift_right_decoder_input

    def _mask_target(self, tgt: torch.Tensor) -> Dict[str, torch.Tensor | np.ndarray | int]:
        mbart_special_tokens = set(self.tokenizer.all_special_ids)
        mask_id = self.tokenizer.mask_token_id

        # At least one token should be masked
        min_masks = 1

        # Build the target input and labels
        tgt_input = tgt.new([token for token in tgt.squeeze(0).tolist() if token not in mbart_special_tokens])
        tgt_labels = tgt.new([mask_id] * len(tgt_input))
        removed_special_tokens = tgt[:, len(tgt_input):].squeeze(0)
        if self.train:
            # Sample the indexes to mask
            if min_masks < len(tgt_input):
                sample_size = np.random.randint(min_masks, len(tgt_input))
            else:
                sample_size = len(tgt_input)

            mask_idxs = np.random.choice(len(tgt_input), size=sample_size, replace=False)

            # Mask the decoder inputs
            tgt_input[mask_idxs] = mask_id
            tgt_labels[mask_idxs] = tgt[:, mask_idxs]
        else:
            # At the start of inference we need to mask the entire decoder input
            tgt_input[:] = mask_id
            mask_idxs = np.arange(len(tgt_input) + 1)
            tgt_labels = tgt

        # Compute the number of tokens to predict
        n_masked_tokens = tgt_input.eq(mask_id).sum().item()

        # Append the removed special tokens to the decoder input
        tgt_input = torch.concat([tgt_input, removed_special_tokens]).unsqueeze(0)
        if self.train:
            tgt_labels = torch.concat([tgt_labels, removed_special_tokens]).unsqueeze(0)

        return {"decoder_input_ids": tgt_input, "labels": tgt_labels, "mask_idxs": mask_idxs,
                "n_masks": n_masked_tokens}

    def __getitem__(self, idx):
        sentence_pair = self.dataset[idx]["translation"]
        src_sentence = "<length> " + sentence_pair[self.src_lang]
        tgt_sentence = sentence_pair[self.tgt_lang]
        tokenized_sentences = self.tokenizer(src_sentence, text_target=tgt_sentence, truncation=True,
                                             max_length=self.max_length, padding="max_length",
                                             add_special_tokens=self.special_tokens, return_tensors="pt")
        input_ids = tokenized_sentences["input_ids"]
        labels = tokenized_sentences["labels"]
        masked_target = self._mask_target(labels)
        decoder_input_ids = masked_target["decoder_input_ids"]
        if self.shift_right and self.special_tokens:
            lang_code_id = self.tokenizer.lang_code_to_id[self.tgt_supported_language]
            decoder_input_ids = shift_tokens_right(decoder_input_ids, self.tokenizer.pad_token_id, lang_code_id)

        return {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids, "labels": masked_target["labels"],
                "mask_idxs": masked_target["mask_idxs"], "n_masks": masked_target["n_masks"]}
