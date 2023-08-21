from typing import Tuple

import torch
from torch.functional import F
from torchmetrics import MeanMetric

from continualnat.models.core.transformer_nat_core import TransformerNATCore
from continualnat.models.glat.config_glat import GLATConfig
from continualnat.models.glat.glat_utils import GlancingSampler, LambdaScheduler
from continualnat.utils.masks import create_masks, create_encoder_mask, create_decoder_mask
from continualnat.utils.models import init_bert_weights


class GLAT(TransformerNATCore):

    def __init__(self, config: GLATConfig) -> None:
        """
        The Glancing Transformer (GLAT) from Qian et al. https://arxiv.org/pdf/2008.07905.pdf, a
        non-autoregressive model that use GLM (Glancing Language Model) to learn word interdependecy. The model's
        architecture is the same as the FT-NAT from Gu et al. https://arxiv.org/pdf/1711.02281.pdf, but it uses a
        <length> token to predict the target sentences' lengths as Ghazvininejad et al.
        https://arxiv.org/pdf/1904.09324.pdf did for the CMLM model.
        """
        super().__init__(config)

        # Scheduler and sampler used by the glancing strategy
        self.lambda_scheduler = LambdaScheduler()
        self.glancing_sampler = GlancingSampler()

        # Use BERT weight initialization
        self.apply(init_bert_weights)

        # Train and validation losses
        self.train_metrics["glat_loss"] = MeanMetric()

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                e_mask: torch.Tensor = None,
                d_mask: torch.Tensor = None,
                src_lengths: torch.Tensor = None,
                tgt_lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process source and target sequences.
        """
        if not self._check_length_token(src_input):
            raise ValueError("The token <length> is not used by one or more tokenized sentence, the model needs such"
                             "token to predict the target lengths.")

        # Compute the source and target lengths if both are not passed
        if (src_lengths is None) ^ (tgt_lengths is None):
            raise ValueError("One between the source and target lengths is None while the other is defined.")
        elif src_lengths is None:
            src_lengths = e_mask.sum(dim=-1)
            src_lengths -= 2 if self.length_token_id is None else 3
            tgt_lengths = d_mask.sum(dim=-1)[:, 0].unsqueeze(-1) - 2

        # Embeddings and positional encoding
        src_embeddings = self.embedding(src_input)  # (bsz, src_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)

        # Encoder
        e_output = self.encoder(src_embeddings, e_mask)

        # Pooling for target lengths
        pooler_inputs, e_output, e_mask = self._define_pooler_inputs(e_output, e_mask)
        predicted_lengths = self.pooler(**pooler_inputs)  # (bsz, src_len)

        # Decoder inputs copy mechanism
        if self.decoder_inputs_copy is not None:
            # Compute the eos and target language tokens embeddings
            tgt_lang_token_id = tgt_input[0, tgt_lengths[0] + 1]  # we assume that the target token comes before any pad
            eos_token_tensor = torch.tensor(self.eos_token_id, device=tgt_lang_token_id.device).unsqueeze(0)
            eos_tgt_lang = torch.cat([eos_token_tensor, tgt_lang_token_id], dim=-1).unsqueeze(0)
            eos_tgt_lang_embeddings = self.embedding(eos_tgt_lang)  # (1, 2, d_model)
            eos_tgt_lang_embeddings = self.positional_encoder(eos_tgt_lang_embeddings * self.embedding_scale)

            # Copy the encoder output or source embeddings without taking into account the special tokens
            tensor_to_copy = e_output if self.tensor_to_copy == "e_output" else src_embeddings
            tensor_to_copy = tensor_to_copy[:, :-2, :]  # do not consider the eos and source language tokens
            new_tgt_embeddings = self._copy_embeddings(tensor_to_copy, d_mask, src_lengths, tgt_lengths)

            # Concat the eos and target language tokens embeddings with the copied target embeddings
            tgt_embeddings = []
            for i, token_embeddings in enumerate(new_tgt_embeddings):
                current_length = tgt_lengths[i]
                concat_embeddings = torch.cat([token_embeddings[:current_length], eos_tgt_lang_embeddings[0],
                                               token_embeddings[current_length:]], dim=0)
                tgt_embeddings.append(concat_embeddings)

            tgt_embeddings = torch.stack(tgt_embeddings, dim=0)
        else:
            tgt_embeddings = self.embedding(tgt_input)  # (bsz, tgt_len, d_model)
            tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)

        # Decoder
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, e_mask)  # (bsz, tgt_len, d_model)

        # Linear output
        output = self.linear_output(d_output)  # (bsz, tgt_len, vocab_size)
        return output, predicted_lengths, e_output, tgt_embeddings

    def on_train_start(self) -> None:
        super().on_train_start()
        self.lambda_scheduler.anneal_steps = self.trainer.estimated_stepping_batches

    def __glancing_strategy(self,
                            labels: torch.Tensor,
                            labels_mask: torch.tensor,
                            tgt_embeddings: torch.Tensor,
                            logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the glancing ratio
        glancing_ratio = self.lambda_scheduler(self.trainer.global_step)

        # Sample the tokens by building a glacing mask (1 if the token is glanced, O otherwise)
        predictions = logits.argmax(dim=-1)  # compute predictions
        glancing_mask = self.glancing_sampler(labels, labels_mask, logits, predictions, glancing_ratio)
        glanced_tokens = glancing_mask.unsqueeze(-1)

        # Compute the embeddings of the predictions
        pred_embeddings = self.embedding(predictions)  # (bsz, tgt_len, d_model)
        pred_embeddings = self.positional_encoder(pred_embeddings * self.embedding_scale)

        # Concatenate the previous target embeddings with the predictions one, based on the glanced postions
        new_tgt_embeddings = glanced_tokens * pred_embeddings + (1 - glanced_tokens) * tgt_embeddings

        # Build the new labels mask that takes into account the glanced positions
        labels_mask_loss = glancing_mask.bool() | ~labels_mask
        labels.masked_fill_(labels_mask_loss, self.pad_token_id)
        return new_tgt_embeddings, labels

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        decoder_input_ids = batch["decoder_input_ids"]
        labels_special_mask = batch["labels_special_mask"]  # (1 if special token, 0 otherwise)
        src_lengths = batch["src_lengths"]  # (bsz, 1)
        tgt_lengths = batch["tgt_lengths"]  # (bsz, 1)

        # Create masks
        e_mask, d_mask = create_masks(input_ids, decoder_input_ids, self.pad_token_id, None)

        # Compute logits and predicted lengths
        logits, predicted_lengths, e_output, tgt_embeddings = self(input_ids, decoder_input_ids, e_mask=e_mask,
                                                                   d_mask=d_mask, src_lengths=src_lengths,
                                                                   tgt_lengths=tgt_lengths)

        # Glancing strategy
        tgt_embeddings, labels = self.__glancing_strategy(labels, ~labels_special_mask.bool(), tgt_embeddings, logits)

        # Compute the new logits
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, e_mask)  # (bsz, tgt_len, d_model)
        logits = self.linear_output(d_output)  # (bsz, tgt_len, vocab_size)

        # Compute loss only on the non-glanced tokens
        loss, logits_loss, lengths_loss = self.compute_loss(logits, labels, predicted_lengths, tgt_lengths)

        # Update train metrics
        self.train_metrics["train_loss"].update(loss.item())
        self.train_metrics["glat_loss"].update(logits_loss)
        self.train_metrics["lengths_loss"].update(lengths_loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        references = batch["references"]
        input_ids = batch["input_ids"]

        # Compute translations
        tokenizer, lang_pair, tgt_lang = self._val_tokenizer_tgtlang(dataloader_idx)
        translation = self.generate(input_ids, tokenizer.lang_code_to_id[tgt_lang])
        predictions = tokenizer.batch_decode(translation, skip_special_tokens=True)

        # Update the BLEU metric internal parameters
        self.val_metrics[f"BLEU_{lang_pair}"].update(predictions, references)

    def generate(self,
                 input_ids: torch.Tensor,
                 tgt_lang_token_id: int,
                 length_beam_size: int = 5) -> torch.Tensor:
        """
        Generate tokens during inference by using the mask-predict algorithm by Ghazvininejad et al.
        https://arxiv.org/pdf/1904.09324.pdf.
        :param input_ids: the tokenized source sentence.
        :param tgt_lang_token_id: the target language token id, if none is passed, then no token will be appended at the
            end of the target tokens (default=None).
        :param length_beam_size: the number of top lengths to consider for each sentence, akin to the beam size of
            the beam search (default=5).
        :return: tokenized translation of source sentence.
        """
        if length_beam_size < 1:
            raise ValueError("The number of lengths to consider for each sentence must be at least 1.")

        if not self._check_length_token(input_ids):
            raise ValueError("You are not using the <length> token at the start of the source sentences,"
                             "the model can not predict the target lengths.")

        if tgt_lang_token_id is None:
            raise ValueError("You should define the target language token id.")

        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Compute encodings
        e_mask = create_encoder_mask(input_ids, self.pad_token_id)
        encodings = self.encode(input_ids, e_mask)

        # Predict the best length_beam_size lengths for each sentence
        length_beams = self.predict_target_length(encodings, length_beam_size, e_mask)
        length_beams[length_beams < 2] = 2

        # Compute the largest length and the number of non-pad tokens
        '''max_length = length_beams.max().item()
        non_pad_tokens = max_length + 2

        # Build the length mask
        length_mask = torch.triu(input_ids.new(non_pad_tokens, non_pad_tokens).fill_(1).long(), 1)
        length_mask = torch.stack([length_mask[length_beams[batch] - 1] for batch in range(batch_size)], dim=0)

        # Initialize target tokens
        tgt_tokens = input_ids.new(batch_size, length_beam_size, non_pad_tokens).fill_(self.mask_token_id)
        tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * self.pad_token_id
        for i, lengths in enumerate(length_beams):
            lengths = lengths + torch.arange(0, length_beam_size * non_pad_tokens, non_pad_tokens).to(device)

            # Add end of sequence token
            tgt_tokens[i].view(-1)[lengths] = self.eos_token_id

            # Add target language token if passed
            if tgt_lang_token_id is not None:
                tgt_tokens[i].view(-1)[lengths + 1] = tgt_lang_token_id

        tgt_tokens = tgt_tokens.view(batch_size * length_beam_size, non_pad_tokens)
        d_mask = create_decoder_mask(tgt_tokens, self.pad_token_id, None)'''

        # Duplicate encoder's output (without taking into account the encodings of the <length> token) and padding mask
        # to match the number of length beams
        start_token = 0 if self.length_token_id is None else 1
        duplicated_encodings = encodings[:, start_token:].repeat_interleave(length_beam_size, dim=0)
        duplicated_e_mask = e_mask[:, :, start_token:].repeat_interleave(length_beam_size, dim=0)

        # Compute the translation tokens and their probabilities
        output = self.decode(tgt_tokens, duplicated_encodings, d_mask=d_mask, e_mask=duplicated_e_mask)
        logits = F.log_softmax(output, dim=-1)
        log_probabilities, translation_tokens = logits.max(dim=-1)

        # Reshape hypotheses and their log probabilities
        hyps = translation_tokens.view(batch_size, length_beam_size, translation_tokens.size(-1))
        log_probabilities = log_probabilities.view(batch_size, length_beam_size)

        # Compute the best lengths in terms of log probabilities
        avg_log_prob = log_probabilities / length_beams.float()
        best_lengths = avg_log_prob.max(-1)[1]

        # Retrieve best hypotheses
        output = torch.stack([hyps[batch, length, :] for batch, length in enumerate(best_lengths)], dim=0)
        return output
