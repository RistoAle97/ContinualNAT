from typing import Tuple

import torch
from torch.functional import F
from torchmetrics import MeanMetric

from continualnat.models.core.transformer_nat_core import TransformerNATCore
from continualnat.models.glat.config_glat import GLATConfig
from continualnat.utils.glancing_utils import GlancingSampler, LambdaScheduler
from continualnat.utils.masks import create_masks, create_encoder_mask, create_padding_mask_from_lengths
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

        # src_lengths += 1  # consider the eos token too

        # Embeddings and positional encoding
        src_embeddings_no_pos_encoding = self.embedding(src_input)  # (bsz, src_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings_no_pos_encoding * self.embedding_scale)

        # Encoder
        e_output = self.encoder(src_embeddings, e_mask)  # (bsz, src_len, d_model)

        # Pooling for target lengths
        pooler_inputs, e_output, e_mask = self._define_pooler_inputs(e_output, e_mask)
        predicted_lengths = self.pooler(**pooler_inputs)  # (bsz, src_len)

        # Copy the source embeddings or the encoder ouptut
        tensor_to_copy = torch.clone(e_output if self.tensor_to_copy == "e_output" else src_embeddings_no_pos_encoding)
        tensor_to_copy = tensor_to_copy[:, :src_lengths.max(), :]
        new_tgt_embeddings = self._copy_embeddings(tensor_to_copy, src_lengths, tgt_lengths)

        # Positional encoding for the copied target embeddings
        # pos_encoded_tgt_input = self.positional_encoder.compute_positions(tgt_input)
        # tgt_embeddings = self.positional_encoder.dropout(new_tgt_embeddings + pos_encoded_tgt_input)
        tgt_embeddings = self.positional_encoder(new_tgt_embeddings * self.embedding_scale)

        # Decoder
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, e_mask)  # (bsz, tgt_len, d_model)

        # Linear output
        output = self.linear_output(d_output)  # (bsz, tgt_len, vocab_size)
        return output, predicted_lengths, e_output, tgt_embeddings  # new_tgt_embeddings

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

        # Concatenate the previous target embeddings with the predictions', based on the glanced postions
        new_tgt_embeddings = glanced_tokens * pred_embeddings + (1 - glanced_tokens) * tgt_embeddings

        # Build the new labels mask that takes into account the glanced positions
        labels.masked_fill_(glancing_mask.bool() | ~labels_mask, self.pad_token_id)
        return new_tgt_embeddings, labels

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        decoder_input_ids = batch["decoder_input_ids"]
        labels_special_mask = batch["labels_special_mask"]  # (1 if special token, 0 otherwise)
        src_lengths = batch["src_lengths"]  # (bsz, 1)
        tgt_lengths = batch["tgt_lengths"]  # (bsz, 1)
        max_tgt_length = tgt_lengths.max()

        # Create masks
        e_mask, d_mask = create_masks(input_ids, decoder_input_ids[:, :max_tgt_length], self.pad_token_id, None)

        # Compute logits and predicted lengths
        logits, predicted_lengths, e_output, tgt_embeddings = self(input_ids, decoder_input_ids, e_mask=e_mask,
                                                                   d_mask=d_mask, src_lengths=src_lengths,
                                                                   tgt_lengths=tgt_lengths)

        # Glancing strategy
        labels = labels[:, :max_tgt_length]
        labels_special_mask = labels_special_mask[:, :max_tgt_length]
        tgt_embeddings, labels = self.__glancing_strategy(labels, ~labels_special_mask.bool(), tgt_embeddings, logits)

        # Compute the new logits
        # tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, e_mask)  # (bsz, tgt_len, d_model)
        logits = self.linear_output(d_output)  # (bsz, tgt_len, vocab_size)

        # Compute loss only on the non-glanced tokens
        loss, logits_loss, lengths_loss = self.compute_loss(logits, labels, predicted_lengths, tgt_lengths)

        # Update train metrics
        self.train_metrics["train_loss"].update(loss.item())
        self.train_metrics["glat_loss"].update(logits_loss)
        self.train_metrics["lengths_loss"].update(lengths_loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        batches = self.trainer.log_every_n_steps * self.trainer.accumulate_grad_batches
        if ((batch_idx == 0 and self.trainer.current_epoch == 0) or
                (batch_idx + 1) * (self.trainer.current_epoch + 1) % batches == 0):
            self.log("Lambda schedule", self.lambda_scheduler.last_ratio)

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
        bsz = input_ids.shape[0]
        # device = input_ids.device

        # Compute lengths of the input ids
        src_lengths = torch.sum(input_ids.ne(self.pad_token_id), dim=-1)
        if tgt_lang_token_id not in input_ids[0]:
            # Need to put the target language token inside the input ids
            new_input_ids = []
            tgt_lang_token = torch.tensor(tgt_lang_token_id).unsqueeze(0)
            for i, tokenized_sentence in enumerate(input_ids):
                new_tokenized_sentence = torch.cat([tokenized_sentence[:src_lengths[i]], tgt_lang_token,
                                                    tokenized_sentence[src_lengths[i]:]], dim=0)
                new_input_ids.append(new_tokenized_sentence)

            input_ids = torch.stack(new_input_ids)

        src_lengths = src_lengths.unsqueeze(-1) - 2  # (bsz, 1)
        # src_lengths = src_lengths.unsqueeze(-1) - 1

        # Compute encodings
        e_mask = create_encoder_mask(input_ids, self.pad_token_id)
        src_embeddings_no_pos_encoding = self.embedding(input_ids)
        src_embeddings = self.positional_encoder(src_embeddings_no_pos_encoding * self.embedding_scale)
        encodings = self.encoder(src_embeddings, e_mask)

        # Duplicate encoder's output (without taking into account the encodings of the <length> token) and padding mask
        # to match the number of length beams
        start_token = 0 if self.length_token_id is None else 1
        duplicated_encodings = encodings[:, start_token:].repeat_interleave(length_beam_size, dim=0)
        duplicated_e_mask = e_mask[:, :, start_token:].repeat_interleave(length_beam_size, dim=0)

        # Predict the best length_beam_size lengths for each sentence
        length_beams = self.predict_target_length(encodings, length_beam_size, e_mask)
        length_beams[length_beams < 2] = 2
        tgt_lengths = length_beams.view(-1, 1)

        # Decide which tensor to copy between the source embeddings and the encodings
        tensor_to_copy = torch.clone(encodings if self.tensor_to_copy == "e_output" else src_embeddings_no_pos_encoding)
        tensor_to_copy = tensor_to_copy[:, :src_lengths.max(), :]
        tensor_to_copy = tensor_to_copy.repeat_interleave(length_beam_size, dim=0)

        # Create the decoder mask
        d_mask = create_padding_mask_from_lengths(tgt_lengths, True)

        # Copy the source embeddings or encodings
        src_lengths = src_lengths.repeat_interleave(length_beam_size, dim=0)
        new_tgt_embeddings = self._copy_embeddings(tensor_to_copy, src_lengths, tgt_lengths)

        # Positional encoding for the new target embeddings
        tgt_embeddings = self.positional_encoder(new_tgt_embeddings * self.embedding_scale)

        # Compute the translation tokens and their probabilities
        d_output = self.decoder(tgt_embeddings, duplicated_encodings, d_mask=d_mask, e_mask=duplicated_e_mask)
        output = self.linear_output(d_output)
        logits = F.log_softmax(output, dim=-1)
        log_probabilities, translation_tokens = logits.max(dim=-1)

        # Insert pad tokens to the masked positions and insert zeros to the respective log probabilities
        d_pad_mask = torch.clone(~d_mask[:, 0, :])
        '''for i, tgt_length in enumerate(tgt_lengths):
            translation_tokens[i, tgt_length - 1] = self.pad_token_id
            log_probabilities[i, tgt_length - 1] = 0.0'''

        translation_tokens.masked_fill_(d_pad_mask, self.pad_token_id)
        log_probabilities.masked_fill_(d_pad_mask, 0.0)

        # Reshape hypotheses and their log probabilities
        hyps = translation_tokens.view(bsz, length_beam_size, translation_tokens.size(-1))
        log_probabilities = log_probabilities.sum(dim=-1).view(bsz, length_beam_size)

        # Compute the best lengths in terms of log probabilities
        avg_log_prob = log_probabilities / length_beams.float()
        best_lengths = avg_log_prob.max(-1)[1]

        # Retrieve best hypotheses
        output = torch.stack([hyps[batch, length, :] for batch, length in enumerate(best_lengths)], dim=0)
        return output
