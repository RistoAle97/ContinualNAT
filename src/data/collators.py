import torch
import random
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy, TensorType
from src.utils import shift_tokens_right
from typing import Dict, Union


class BatchCollator:

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 truncation: bool = True,
                 max_length: Union[int, None] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 add_special_tokens: bool = True,
                 return_tensors: Union[str, TensorType, None] = "pt",
                 use_language_tokens: bool = True,
                 is_mlm: bool = False,
                 use_cls_token: bool = False) -> None:
        """
        Standard collator, its work consists in tokenizing the source and target sentences, batching them and creating
        the decoder inputs.
        :param tokenizer: the tokenizer used by the collator when called.
        :param truncation: whether to apply truncation during the tokenization (default=True).
        :param max_length: maximum allowed length fot the tokenized sentences (default=None)
        :param padding: the padding strategy to apply during the tokenization (defualt=True).
        :param add_special_tokens: whether to use special tokens during the tokenization (default=True)-
        :param return_tensors: type of tensors from the tokenizer (default="pt").
        :param use_language_tokens: whether to use language tokens by the tokenizer (default=True).
        :param is_mlm: whether the collator is used for a masked language model (MLM), it False then we're dealing with
            a causal model and the labels must be shifted to the right in order to create the decoder inputs
            (default=False).
        :param use_cls_token: whether to add the cls token at the beginnning of the source sentences (default=False).
        """
        self.tokenizer = tokenizer
        self.use_language_tokens = use_language_tokens
        self.is_mlm = is_mlm
        self.use_cls_token = use_cls_token
        self.collator_state = {"truncation": truncation, "max_length": max_length, "padding": padding,
                               "add_special_tokens": add_special_tokens, "return_tensors": return_tensors}

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        # Build and tokenize the batches
        src_sentences = [sentence_pair["src_sentence"] for sentence_pair in batch]
        if self.use_cls_token:
            src_sentences = [self.tokenizer.cls_token + " " + src_sentence for src_sentence in src_sentences]

        tgt_sentences = [sentence_pair["tgt_sentence"] for sentence_pair in batch]
        input_ids_batch = self.tokenizer(src_sentences, **self.collator_state)["input_ids"]
        labels_batch = self.tokenizer(text_target=tgt_sentences, **self.collator_state)["input_ids"]

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
        if not self.is_mlm:
            if self.use_language_tokens and tokenizer_has_lang_codes and labels_batch[0, 0]:
                tgt_lang_token_id = self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
                starts_with_tgt_lang_token = labels_batch[0, 0].eq(tgt_lang_token_id)
                if starts_with_tgt_lang_token:
                    # If the labels start with the tgt lang token we perform the usual shift
                    decoder_input_ids_batch = labels_batch[:, :-1]
                    labels_batch = labels_batch[:, 1:]
                else:
                    # Otherwise we move the tgt lang token to the first position in a MBart fashion
                    decoder_input_ids_batch = shift_tokens_right(labels_batch, self.tokenizer.pad_token_id)
            else:
                # The usual shift seen for causal language models
                decoder_input_ids_batch = labels_batch[:, :-1]
                labels_batch = labels_batch[:, 1:]
        else:
            # This is applied for masked language models such as CMLM
            decoder_input_ids_batch = labels_batch.detach().clone()

        return {"input_ids": input_ids_batch, "labels": labels_batch, "decoder_input_ids": decoder_input_ids_batch}


class BatchCollatorCMLM(BatchCollator):

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 truncation: bool = True,
                 max_length: Union[int, None] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 add_special_tokens: bool = True,
                 return_tensors: Union[str, TensorType, None] = "pt",
                 use_language_tokens: bool = True,
                 train: bool = False) -> None:
        """
        Variation of the standard batch collator, used mainly for the CMLM model. At training time, the
        decoder inputs (except for the pad and language tokens) are masked by a random number in
        [1, seq_len - n_special_tokens], the labels are then padded where the masks are placed. At inference time,
        all the decoder inputs (except, again, the pad and language tokens) are masked.
        :param tokenizer: the tokenizer used by the collator when called.
        :param truncation: whether to apply truncation during the tokenization (default=True).
        :param max_length: maximum allowed length fot the tokenized sentences (default=None)
        :param padding: the padding strategy to apply during the tokenization (defualt=True).
        :param add_special_tokens: whether to use special tokens during the tokenization (default=True)-
        :param return_tensors: type of tensors from the tokenizer (default="pt").
        :param train: whether the collator is used during training (default=False).
        """
        super().__init__(tokenizer, truncation, max_length, padding, add_special_tokens, return_tensors,
                         use_language_tokens, True, True)
        # Parameters
        self.train = train

    def __mask_target(self, labels: torch.Tensor, decoder_input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = labels.size()

        # Retrieve all the special tokens from the tokenizer and its mask id
        if self.tokenizer.mask_token_id is None:
            raise ValueError("For the CMLM model you should use a tokenizer that has a mask token defined.")

        mask_token_id = self.tokenizer.mask_token_id

        # Build the special tokens mask and get how many of them are there for each sentence
        special_tokens_masks = [self.tokenizer.get_special_tokens_mask(sentence, already_has_special_tokens=True)
                                for sentence in labels]
        special_tokens_masks = torch.tensor(special_tokens_masks)  # 1 if special token, 0 otherwise
        n_special_tokens = torch.sum(special_tokens_masks, dim=-1)

        # At least one token per each sentence should be masked
        min_masks = 1

        # Keep the indexes of those tokens that can be masked and the number of such tokens
        maskable_tokens_idxs = [(mask == 0).nonzero(as_tuple=True)[0].tolist() for mask in special_tokens_masks]
        n_maskable_tokens = seq_len - n_special_tokens

        # Mask tokens loop
        if self.train:
            # At training time we mask out a number of tokens in [1, seq_len - n_special_tokens] from the decoder inputs
            labels = labels.new(batch_size, seq_len).fill_(self.tokenizer.pad_token_id)
            for i, max_tokens_to_mask in enumerate(n_maskable_tokens):
                if max_tokens_to_mask > 0:
                    # Sample the number of tokens to mask with a uniform distribution
                    sample_size = np.random.randint(min_masks, max_tokens_to_mask + 1)

                    # Sample the idxs to mask
                    masks = random.sample(maskable_tokens_idxs[i], sample_size)

                    # Mask the decoder inputs
                    labels[i, masks] = decoder_input_ids[i, masks]
                    decoder_input_ids[i, masks] = mask_token_id
                else:
                    labels[i] = decoder_input_ids[i]
        else:
            # At inference time we mask the entire decoder inputs
            for i, maskable_tokens in enumerate(maskable_tokens_idxs):
                decoder_input_ids[i, maskable_tokens] = mask_token_id

        return {"labels": labels, "decoder_input_ids": decoder_input_ids, "lengths": n_maskable_tokens}

    def __call__(self, batch):
        tokenized_batch = super().__call__(batch)
        input_ids = tokenized_batch["input_ids"]
        masked_target = self.__mask_target(tokenized_batch["labels"], tokenized_batch["decoder_input_ids"])
        return {"input_ids": input_ids, "labels": masked_target["labels"],
                "decoder_input_ids": masked_target["decoder_input_ids"], "target_lengths": masked_target["lengths"]}
