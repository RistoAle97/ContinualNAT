import torch
import random
import numpy as np
from src.utils import shift_lang_token_right
from typing import Dict, List, Union


class BatchCollator:

    def __init__(self,
                 is_mlm: bool = False,
                 shift_lang_token: bool = False,
                 return_special_tokens_mask: bool = False,
                 pad_token_id: int = 1) -> None:
        """
        Standard collator, its work consists in batching the source and target sentences and creating
        the decoder inputs.
        :param is_mlm: whether the collator is used for a masked language model (MLM), if False then we're dealing with
            a causal model and the labels must be shifted to the right in order to create the decoder inputs
            (default=False).
        :param shift_lang_token: whether to move the lang token at the beginnning of the source sentences
            (default=False).
        :param return_special_tokens_mask: whether to return the special tokens mask for the labels (default=False).
        :param pad_token_id: the pad token id (default=1).
        """
        self.is_mlm = is_mlm
        self.shift_lang_token = shift_lang_token
        self.return_special_tokens_mask = return_special_tokens_mask
        self.pad_token_id = pad_token_id

    def __call__(self, batch) -> Dict[str, Union[torch.Tensor, List[List[str]]]]:
        # Put all the tokenized source and target sentences together and save the max length found for both
        src_max_length = 0
        tgt_max_length = 0
        src_tokenized_sentences = []
        tgt_tokenized_sentences = []
        labels_special_tokens_masks = []
        references: List[List[str]] = []
        for sentence_pair in batch:
            tokenized_src = sentence_pair["input_ids"]
            tokenized_tgt = sentence_pair["labels"]
            src_tokenized_sentences.append(tokenized_src)
            tgt_tokenized_sentences.append(tokenized_tgt)
            src_max_length = max(src_max_length, tokenized_src.size(-1))
            tgt_max_length = max(tgt_max_length, tokenized_tgt.size(-1))
            references.append([sentence_pair["reference"]])
            labels_special_tokens_masks.append(sentence_pair["special_mask_labels"])

        # Pad the tensors and batchify them
        input_ids = [torch.cat([src, src.new(1, src_max_length - src.size(-1)).fill_(self.pad_token_id)], dim=-1)
                     for src in src_tokenized_sentences]
        labels = [torch.cat([tgt, tgt.new(1, tgt_max_length - tgt.size(-1)).fill_(self.pad_token_id)], dim=-1)
                  for tgt in tgt_tokenized_sentences]
        input_ids = torch.stack(input_ids, dim=0).squeeze(1)
        labels = torch.stack(labels, dim=0).squeeze(1)

        # Create decoder input ids
        if not self.is_mlm:
            if self.shift_lang_token:
                # Move the tgt lang token to the first position in a MBart fashion
                decoder_input_ids = shift_lang_token_right(labels, self.pad_token_id)
            else:
                # The usual shift seen for causal language models
                decoder_input_ids = labels[:, :-1]
                labels = labels[:, 1:]
        else:
            # This is applied for masked language models such as CMLM
            decoder_input_ids = labels.detach().clone()

        special_tokens_mask = None
        if self.return_special_tokens_mask:
            special_tokens_mask = [torch.cat([mask, mask.new(1, tgt_max_length - mask.size(-1)).fill_(1)], dim=-1)
                                   for mask in labels_special_tokens_masks]
            special_tokens_mask = torch.stack(special_tokens_mask, dim=0).squeeze(1)

        return {"input_ids": input_ids, "labels": labels, "decoder_input_ids": decoder_input_ids,
                "special_tokens_mask_labels": special_tokens_mask, "references": references}


class BatchCollatorCMLM(BatchCollator):

    def __init__(self, pad_token_id: int = 1, mask_token_id: int = 5, train: bool = False) -> None:
        """
        Variation of the standard batch collator, used mainly for the CMLM model. At training time, the
        decoder inputs (except for special tokens like pad and lang) are masked by a random number in
        [1, seq_len - n_special_tokens], the labels are then padded where the masks are placed. At inference time,
        all the decoder inputs (except, again, the special tokens) are masked.
        :param pad_token_id: the pad token id (default=1).
        :param mask_token_id: the mask token id (default=5).
        :param train: whether the collator is used during training (default=False).
        """
        super().__init__(True, False, True, pad_token_id)
        # Parameters
        self.mask_token_id = mask_token_id
        self.train = train

    def __mask_target(self,
                      labels: torch.Tensor,
                      decoder_input_ids: torch.Tensor,
                      special_tokens_mask_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = labels.size()

        # Compute the number of special tokens for each sentence
        n_special_tokens = torch.sum(special_tokens_mask_labels, dim=-1)

        # At least one token per each sentence should be masked
        min_masks = 1

        # Keep the indexes of those tokens that can be masked and the number of such tokens
        maskable_tokens_idxs = [(mask == 0).nonzero(as_tuple=True)[0].tolist() for mask in special_tokens_mask_labels]
        n_maskable_tokens = seq_len - n_special_tokens

        # Mask tokens loop
        if self.train:
            # At training time we mask out a number of tokens in [1, seq_len - n_special_tokens] from the decoder inputs
            labels = labels.new(batch_size, seq_len).fill_(self.pad_token_id)
            for i, max_tokens_to_mask in enumerate(n_maskable_tokens):
                if max_tokens_to_mask > 0:
                    # Sample the number of tokens to mask with a uniform distribution
                    sample_size = np.random.randint(min_masks, max_tokens_to_mask + 1)

                    # Sample the idxs to mask
                    masks = random.sample(maskable_tokens_idxs[i], sample_size)

                    # Mask the decoder inputs
                    labels[i, masks] = decoder_input_ids[i, masks]
                    decoder_input_ids[i, masks] = self.mask_token_id
                else:
                    labels[i] = decoder_input_ids[i]
        else:
            # At inference time we mask the entire decoder inputs
            for i, maskable_tokens in enumerate(maskable_tokens_idxs):
                decoder_input_ids[i, maskable_tokens] = self.mask_token_id

        return {"labels": labels, "decoder_input_ids": decoder_input_ids, "lengths": n_maskable_tokens}

    def __call__(self, batch) -> Dict[str, Union[torch.Tensor, List[List[str]]]]:
        tokenized_batch = super().__call__(batch)
        input_ids = tokenized_batch["input_ids"]
        masked_target = self.__mask_target(tokenized_batch["labels"], tokenized_batch["decoder_input_ids"],
                                           tokenized_batch["special_tokens_mask_labels"])
        return {"input_ids": input_ids, "references": tokenized_batch["references"], "labels": masked_target["labels"],
                "decoder_input_ids": masked_target["decoder_input_ids"], "target_lengths": masked_target["lengths"]}
