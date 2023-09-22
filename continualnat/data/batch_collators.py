from typing import Dict, List, Union

import numpy as np
import torch

from continualnat.utils.utils import shift_lang_token_right


class BatchCollator:

    def __init__(self,
                 is_mlm: bool = False,
                 shift_lang_token: bool = False,
                 return_special_masks: bool = False,
                 return_lengths: bool = False,
                 pad_token_id: int = 1,
                 mask_token_id: int = 5,
                 p_masking: Union[float, str] = 0.0) -> None:
        """
        Standard collator, its work consists in batching the source and target sentences and creating
        the decoder inputs.
        :param is_mlm: whether the collator is used for a masked language model (MLM). If False, then we're dealing with
            a causal model and the labels must be shifted to the right in order to create the decoder inputs
            (default=False).
        :param shift_lang_token: whether to move the lang token at the beginning of the source sentences
            (default=False).
        :param return_special_masks: whether to return the special tokens mask for the both input ids and labels
            (default=False).
        :param return_lengths: whether to return the source and target lengths, keep in mind that they do not take into
            account special tokens.
        :param pad_token_id: the pad token id (default=1).
        :param mask_token_id: the mask token id (default=5).
        :param p_masking: the percentage of decoder input ids to mask, can also be "random" (default=0.0).
        """
        self.is_mlm = is_mlm
        self.shift_lang_token = shift_lang_token
        self.return_special_masks = return_special_masks
        self.return_lengths = return_lengths
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        # Sanity checks for the mask token id and the mask percentage
        if isinstance(p_masking, float):
            if not 0 <= p_masking <= 1:
                raise ValueError("The masking probability must be a float between 0 and 1.")

            self.p_masking_is_defined = p_masking > 0.0
            if mask_token_id is None:
                raise ValueError("You defined a value for p_masking but you have passed None as mask token id.")

        elif p_masking != "random":
            raise ValueError("The only accepted string value for p_masking is \"random\".")
        else:
            self.p_masking_is_defined = True

        if self.p_masking_is_defined and mask_token_id is None:
            raise ValueError("You defined a value for p_masking but you have passed None as mask token id.")

        self.p_masking = p_masking

    def __mask_decoder_input_ids(self,
                                 labels: torch.Tensor,
                                 decoder_input_ids: torch.Tensor,
                                 labels_special_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = labels.size()

        # Compute the number of special tokens for each sentence
        n_special_tokens = torch.sum(labels_special_mask, dim=-1)

        # At least one token per each sentence should be masked
        min_masks = 1

        # Keep the indexes of those tokens that can be masked and the number of such tokens
        maskable_tokens_idxs = [(mask == 0).nonzero(as_tuple=True)[0].tolist() for mask in labels_special_mask]
        n_maskable_tokens = seq_len - n_special_tokens

        # Mask tokens loop
        if (isinstance(self.p_masking, float) and self.p_masking != 1.0) or isinstance(self.p_masking, str):
            # At training time we mask out a number of tokens in [1, seq_len - n_special_tokens] from the decoder inputs
            labels = labels.new(batch_size, seq_len).fill_(self.pad_token_id)
            np_generator = np.random.default_rng()
            for i, max_tokens_to_mask in enumerate(n_maskable_tokens):
                if max_tokens_to_mask > 0:
                    # Sample the number of tokens to mask with a uniform distribution
                    if self.p_masking == "random":
                        sample_size = np_generator.integers(min_masks, max_tokens_to_mask + 1)
                    else:
                        sample_size = max_tokens_to_mask * self.p_masking
                        sample_size = min_masks if sample_size == 0 else sample_size

                    # Sample the idxs to mask
                    masks = np_generator.choice(maskable_tokens_idxs[i], sample_size, replace=False)

                    # Mask the decoder inputs
                    labels[i, masks] = decoder_input_ids[i, masks]
                    decoder_input_ids[i, masks] = self.mask_token_id
                else:
                    labels[i] = decoder_input_ids[i]
        else:
            # At inference time we mask the entire decoder inputs
            for i, maskable_tokens in enumerate(maskable_tokens_idxs):
                decoder_input_ids[i, maskable_tokens] = self.mask_token_id

        return {"labels": labels, "decoder_input_ids": decoder_input_ids}

    def __call__(self, batch) -> Dict[str, Union[torch.Tensor, List[List[str]]]]:
        # Put all the tokenized source and target sentences together
        src_max_length = 0
        tgt_max_length = 0
        src_tokenized_sentences = []
        tgt_tokenized_sentences = []
        input_ids_special_masks = []
        labels_special_masks = []
        references: List[List[str]] = []
        for sentence_pair in batch:
            tokenized_src = sentence_pair["input_ids"]
            tokenized_tgt = sentence_pair["labels"]
            src_tokenized_sentences.append(tokenized_src)
            tgt_tokenized_sentences.append(tokenized_tgt)
            src_max_length = max(src_max_length, tokenized_src.size(-1))
            tgt_max_length = max(tgt_max_length, tokenized_tgt.size(-1))
            references.append([sentence_pair["reference"]])
            input_ids_special_masks.append(sentence_pair["input_ids_special_mask"])
            labels_special_masks.append(sentence_pair["labels_special_mask"])

        # Pad the tensors and batchify them
        input_ids = [torch.cat([src, src.new(1, src_max_length - src.size(-1)).fill_(self.pad_token_id)], dim=-1)
                     for src in src_tokenized_sentences]
        labels = [torch.cat([tgt, tgt.new(1, tgt_max_length - tgt.size(-1)).fill_(self.pad_token_id)], dim=-1)
                  for tgt in tgt_tokenized_sentences]
        input_ids = torch.stack(input_ids, dim=0).squeeze(1)  # (bsz, src_max_length)
        labels = torch.stack(labels, dim=0).squeeze(1)  # (bsz, tgt_max_length)

        # Create the decoder input ids
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

        # Compute the special tokens masks
        input_ids_special_mask = None
        labels_special_mask = None
        if self.return_special_masks or self.return_lengths:
            input_ids_special_mask = [torch.cat([mask, mask.new(1, src_max_length - mask.size(-1)).fill_(1)], dim=-1)
                                      for mask in input_ids_special_masks]
            input_ids_special_mask = torch.stack(input_ids_special_mask, dim=0).squeeze(1)  # (bsz, src_max_length)
            labels_special_mask = [torch.cat([mask, mask.new(1, tgt_max_length - mask.size(-1)).fill_(1)], dim=-1)
                                   for mask in labels_special_masks]
            labels_special_mask = torch.stack(labels_special_mask, dim=0).squeeze(1)  # (bsz, tgt_max_length)

            # Pad the special tokens since they should not be predicted by masked language models
            if self.is_mlm:
                labels = (1 - labels_special_mask) * labels + labels_special_mask * self.pad_token_id

        # Compute the source and target lengths, they do not take into account the special tokens
        src_lengths = []
        tgt_lengths = []
        if self.return_lengths:
            src_lengths = torch.sum(input_ids_special_mask.ne(1), dim=-1).unsqueeze(-1)  # (bsz, 1)
            tgt_lengths = torch.sum(labels_special_mask.ne(1), dim=-1).unsqueeze(-1)  # (bsz, 1)

        # Mask the decoder input ids
        if self.p_masking_is_defined:
            masked_target = self.__mask_decoder_input_ids(labels, decoder_input_ids, labels_special_mask)
            labels = masked_target["labels"]
            decoder_input_ids = masked_target["decoder_input_ids"]

        return {"input_ids": input_ids, "labels": labels, "decoder_input_ids": decoder_input_ids,
                "input_ids_special_mask": input_ids_special_mask, "labels_special_mask": labels_special_mask,
                "references": references, "src_lengths": src_lengths, "tgt_lengths": tgt_lengths}
