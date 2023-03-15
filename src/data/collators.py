import torch
import random
import numpy as np
from transformers import PreTrainedTokenizer
from transformers.utils import PaddingStrategy, TensorType
from src import shift_tokens_right
from typing import Dict, List, Union


class BatchCollator:

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 truncation: bool = True,
                 max_length: Union[int, None] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 add_special_tokens: bool = True,
                 return_tensors: Union[str, TensorType, None] = "pt",
                 use_language_tokens: bool = True,
                 shift_labels_right: bool = True) -> None:
        self.tokenizer = tokenizer
        self.use_language_tokens = use_language_tokens
        self.shift_labels_right = shift_labels_right
        self.collator_state = {"truncation": truncation, "max_length": max_length, "padding": padding,
                               "add_special_tokens": add_special_tokens, "return_tensors": return_tensors}

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        # Build and tokenize the batches
        input_ids_batch = [sentence_pair["src_sentence"] for sentence_pair in batch]
        labels_batch = [sentence_pair["tgt_sentence"] for sentence_pair in batch]
        input_ids_batch = self.tokenizer(input_ids_batch, **self.collator_state)["input_ids"]
        labels_batch = self.tokenizer(text_target=labels_batch, **self.collator_state)["input_ids"]

        # Check if the tokenizer support language codes
        tokenizer_has_lang_codes = hasattr(self.tokenizer, "src_lang") and hasattr(self.tokenizer, "tgt_lang")

        # Language tokens are removed if requested
        if not self.use_language_tokens and tokenizer_has_lang_codes:
            pad_token = self.tokenizer.pad_token_id
            src_lang_token = self.tokenizer.lang_code_to_id[self.tokenizer.src_lang]
            tgt_lang_token = self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
            input_ids_batch = torch.where(input_ids_batch == src_lang_token, pad_token, input_ids_batch)
            labels_batch = torch.where(labels_batch == tgt_lang_token, pad_token, labels_batch)

        # Create decoder input ids
        if self.shift_labels_right:
            if self.use_language_tokens and tokenizer_has_lang_codes:
                pad_token_id = self.tokenizer.pad_token_id
                decoder_start_token_id = self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
                decoder_input_ids_batch = shift_tokens_right(labels_batch, pad_token_id, decoder_start_token_id)
            else:
                decoder_input_ids_batch = labels_batch[:, :-1]
                labels_batch = labels_batch[:, 1:]
        else:
            decoder_input_ids_batch = labels_batch

        return {"input_ids": input_ids_batch, "labels": labels_batch, "decoder_input_ids": decoder_input_ids_batch}


class BatchCollatorCMLM(BatchCollator):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 truncation: bool = True,
                 max_length: Union[int, None] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 add_special_tokens: bool = True,
                 return_tensors: Union[str, TensorType, None] = "pt",
                 use_language_tokens: bool = True,
                 train: bool = False) -> None:
        super().__init__(tokenizer, truncation, max_length, padding, add_special_tokens, return_tensors,
                         use_language_tokens)
        self.train = train

    def _mask_target(self, tgt: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[int]]]:
        # Retrieve all the special tokens from the tokenizer and its mask id
        tokenizer_special_tokens = self.tokenizer.all_special_ids
        tokenizer_special_tokens.remove(self.tokenizer.bos_token_id)  # sos can be masked
        tokenizer_special_tokens.remove(self.tokenizer.eos_token_id)  # eos can be masked
        mask_id = self.tokenizer.mask_token_id

        # Initialize decoder inputs and labels
        decoder_input_ids = tgt.detach().clone()
        labels = tgt.detach().clone()

        # At least one token should be masked
        min_masks = 1

        # Compute the length of the target sentences without taking special tokens into account
        n_special_tokens = torch.sum(torch.isin(tgt, torch.tensor(tokenizer_special_tokens)), dim=-1)
        tgt_lengths = tgt.shape[-1] - n_special_tokens

        # Mask tokens loop
        n_masks = []
        mask_idxs = []
        if self.train:
            # At training time we mask out a number of tokens in [1, seq_len - n_special_tokens] from the decoder inputs
            for i, tgt_length in enumerate(tgt_lengths):
                if min_masks < tgt_length:
                    sample_size = np.random.randint(min_masks, tgt_length)
                else:
                    sample_size = min_masks

                n_masks.append(sample_size)
                masks = random.sample(range(tgt_length), sample_size)
                decoder_input_ids[i, masks] = mask_id
                labels[i, :tgt_length] = mask_id
                labels[i, masks] = tgt[i, masks]
                mask_idxs.append(masks)
        else:
            # At inference time we mask the entire decoder inputs
            for i, tgt_length in enumerate(tgt_lengths):
                decoder_input_ids[i, :tgt_length] = mask_id
                mask_idxs.append(list(np.arange(0, tgt_length)))

            labels = tgt
            n_masks = tgt_lengths.tolist()

        return {"decoder_input_ids": decoder_input_ids, "labels": labels, "mask_idxs": mask_idxs,
                "n_masks": n_masks}

    def __call__(self, batch):
        tokenized_batch = super().__call__(batch)
        input_ids, labels = tokenized_batch["input_ids"], tokenized_batch["labels"]
        masked_target = self._mask_target(labels)
        return {"input_ids": input_ids, "decoder_input_ids": masked_target["decoder_input_ids"],
                "labels": masked_target["labels"], "mask_idxs": masked_target["mask_idxs"],
                "n_masks": masked_target["n_masks"]}
