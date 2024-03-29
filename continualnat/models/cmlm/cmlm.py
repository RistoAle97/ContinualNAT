import torch
from torch.functional import F
from torchmetrics import MeanMetric

from continualnat.models.core.transformer_nat_core import TransformerNATCore
from continualnat.models.cmlm.config_cmlm import CMLMConfig
from continualnat.utils.glancing_utils import LambdaScheduler, GlancingSampler
from continualnat.utils.masks import create_masks, create_encoder_mask
from continualnat.utils.models import init_bert_weights


class CMLM(TransformerNATCore):
    def __init__(self, config: CMLMConfig) -> None:
        """
        The Conditional Masked Language Model (CMLM) from Ghazvininejad et al. https://arxiv.org/pdf/1904.09324.pdf, a
        non-autoregressive model whose training is based on BERT by Devlin et al. https://arxiv.org/pdf/1810.04805.pdf
        and uses an iterative decoding strategy called "mask-predict" during inference.
        """
        super().__init__(config)

        # Some common NAT parameters are not used by the CMLM model
        del self.map_copy
        del self.tensor_to_copy
        del self.tau

        # Scheduler and sampler used by the glancing strategy
        self.glat_training = config.glat_training
        if self.glat_training:
            self.lambda_scheduler = None
            self.glancing_sampler = GlancingSampler()

        # Token ids
        self.mask_token_id = config.mask_token_id

        # Use BERT weight initialization
        self.apply(init_bert_weights)

        # Train and validation losses
        self._training_loss = "glat_loss" if self.glat_training else "mlm_loss"
        self.train_metrics[self._training_loss] = MeanMetric()

    def forward(
        self,
        src_input: torch.Tensor,
        tgt_input: torch.Tensor,
        e_mask: torch.Tensor = None,
        d_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process source and target sequences.
        """
        if not self._check_length_token(src_input):
            raise ValueError(
                "The token <length> is not used by one or more tokenized sentence, the model needs such token to"
                "predict the target lengths."
            )

        # Embeddings and positional encoding
        src_embeddings = self.embedding(src_input)  # (bsz, src_len, d_model)
        tgt_embeddings = self.embedding(tgt_input)  # (bsz, src_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)
        tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)

        # Encoder
        e_output = self.encoder(src_embeddings, e_mask)  # (bsz, src_len, d_model)

        # Pooling for target lengths
        pooler_inputs, e_output, e_mask = self._define_pooler_inputs(e_output, e_mask)
        predicted_lengths = self.pooler(**pooler_inputs)  # (bsz, pooler_size)

        # Decoder
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, e_mask)  # (bsz, tgt_len, d_model)

        # Linear output
        output = self.linear_output(d_output)  # (bsz, tgt_len, vocab_size)
        return output, predicted_lengths, e_output, d_output

    def on_train_start(self) -> None:
        super().on_train_start()
        if self.glat_training:
            self.lambda_scheduler = LambdaScheduler(steps=self.trainer.estimated_stepping_batches)

    def __glancing_strategy(
        self,
        labels: torch.Tensor,
        labels_non_special_mask: torch.tensor,
        decoder_input_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute the glancing ratio
        glancing_ratio = self.lambda_scheduler(self.trainer.global_step)

        # Sample the tokens by building a glacing mask (1 if the token is glanced, O otherwise)
        predictions = logits.argmax(dim=-1)  # compute predictions
        glancing_mask = self.glancing_sampler(labels, labels_non_special_mask, logits, predictions, glancing_ratio)

        # The non glanced tokens should be masked, except for the eos, pad and language tokens
        glanced_decoder_input_ids = glancing_mask * labels + (1 - glancing_mask) * self.mask_token_id
        glanced_decoder_input_ids = (
            labels_non_special_mask * glanced_decoder_input_ids + (1 - labels_non_special_mask) * decoder_input_ids
        )

        # Modify the labels such that they take into account the glanced positions
        labels.masked_fill_(glancing_mask.bool() | ~labels_non_special_mask.bool(), self.pad_token_id)
        return glanced_decoder_input_ids, labels

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        decoder_input_ids = batch["decoder_input_ids"]
        labels_special_mask = batch["labels_special_mask"]
        tgt_lengths = batch["tgt_lengths"]

        # Create masks
        e_mask, d_mask = create_masks(input_ids, decoder_input_ids, self.pad_token_id, None)

        # Compute loss
        logits, length_logits, e_output, _ = self(input_ids, decoder_input_ids, e_mask=e_mask, d_mask=d_mask)

        # Employ glancing strategy
        if self.glat_training:
            glanced_decoder_input_ids, labels = self.__glancing_strategy(
                labels, 1 - labels_special_mask, decoder_input_ids, logits
            )

            # Compute the new logits
            logits = self.decode(glanced_decoder_input_ids, e_output, d_mask, e_mask)

        loss, logits_loss, lengths_loss = self.compute_loss(logits, labels, length_logits, tgt_lengths)

        # Update train metrics
        self.train_metrics["train_loss"].update(loss.item())
        self.train_metrics[self._training_loss].update(logits_loss)
        self.train_metrics["lengths_loss"].update(lengths_loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.glat_training:
            if self.batches_seen == 1:
                lambda_metric = {"Lambda schedule": self.lambda_scheduler.start_ratio}
                self.logger.log_metrics(lambda_metric, 0)
            elif self.batches_seen % self.log_every_n_batches == 0:
                self.log("Lambda schedule", self.lambda_scheduler.last_ratio)

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        input_ids = batch["input_ids"]
        references = batch["references"]

        # Compute translations
        tokenizer, lang_pair, tgt_lang = super()._val_tokenizer_tgtlang(dataloader_idx)
        translation, _ = self.generate(input_ids, tokenizer.lang_code_to_id[tgt_lang])
        predictions = tokenizer.batch_decode(translation, skip_special_tokens=True)

        # Update the BLEU metric internal parameters
        self.val_metrics[f"BLEU_{lang_pair}"].update(predictions, references)

    def __mask_predict(
        self,
        encodings: torch.Tensor,
        e_mask: torch.Tensor,
        tgt_input: torch.Tensor,
        iterations: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Parameters
            bsz, seq_len = tgt_input.size()
            device = next(self.parameters()).device

            # Keep track of the tokens generated at each step
            output_at_each_step = tgt_input.detach().clone()

            # Make the first prediction in a fully non-autoregressive way
            d_mask = tgt_input.ne(self.pad_token_id).unsqueeze(1).to(device)  # this mask will never change
            tgt_lengths = seq_len - tgt_input.ne(self.mask_token_id).sum(dim=-1)
            output = self.decode(tgt_input, encodings, d_mask=d_mask, e_mask=e_mask)
            logits = F.log_softmax(output, dim=-1)
            p_tokens, tokens = logits.max(dim=-1)  # tokens probabilites and their ids

            # Non-maskable tokens (such as eos, pad and lang tokens) should not be among those to be predicted
            non_maskable_tokens = torch.zeros(bsz, seq_len, dtype=torch.int)
            non_maskable_tokens[(torch.arange(bsz), tgt_lengths.view(-1))] = 1
            non_maskable_tokens = non_maskable_tokens.cumsum(dim=1)
            non_maskable_tokens = (non_maskable_tokens == 1).view(-1)

            # Take the non-maskable tokens from the target input and set their probabilities to zero in order to not
            # choose them during mask-predict iterations
            tokens.view(-1)[non_maskable_tokens] = tgt_input.view(-1)[non_maskable_tokens]
            p_tokens.view(-1)[non_maskable_tokens] = 0

            # Save the tokens generated from the full non-autoregressive step
            output_at_each_step = torch.cat([output_at_each_step, tokens], dim=-1)

            # Mask-predict iterations
            for i in range(1, iterations):
                # Compute the number of masks per sentence
                n_masks = (tgt_lengths * (1.0 - i / iterations)).int()

                # Compute the indexes of the worst tokens in terms of probability
                masks = [
                    p_tokens[batch, : tgt_lengths[batch]].topk(max(1, n_masks[batch]), largest=False, sorted=False)[1]
                    for batch in range(bsz)
                ]
                masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
                masks = torch.stack(masks, dim=0)

                # Apply the masks
                masks = masks + torch.arange(0, bsz * seq_len, seq_len, device=device).unsqueeze(1)
                tokens.view(-1)[masks.view(-1)] = self.mask_token_id

                # Compute the new tokens and their probabilities
                output = self.decode(tokens, encodings, d_mask=d_mask, e_mask=e_mask)
                logits = F.log_softmax(output, dim=-1)
                new_p_tokens, new_tokens = logits.max(dim=-1)

                # Update the output tokens and probabilities
                p_tokens.view(-1)[masks.view(-1)] = new_p_tokens.view(-1)[masks.view(-1)]
                tokens.view(-1)[masks.view(-1)] = new_tokens.view(-1)[masks.view(-1)]

                # Save the tokens generated at the i-nth step
                output_at_each_step = torch.cat([output_at_each_step, tokens], dim=-1)

            # Sum the log probabilities of the tokens for each sentence
            log_p_tokens = p_tokens.sum(-1)

            # Modify the shape of the tensor containing the tokens for each step
            output_at_each_step = output_at_each_step.view(bsz, iterations + 1, seq_len)
            return tokens, log_p_tokens, output_at_each_step

    def generate(
        self,
        input_ids: torch.Tensor,
        tgt_lang_token_id: int,
        iterations: int = None,
        length_beam_size: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens during inference by using the mask-predict algorithm by Ghazvininejad et al.
        https://arxiv.org/pdf/1904.09324.pdf.
        :param input_ids: the tokenized source sentence.
        :param tgt_lang_token_id: the target language token id, if none is passed, then no token will be appended at the
            end of the target tokens (default=None).
        :param iterations: the number of iterations of the mask-predict. If its value is equal to 1, then the decoding
            will be purely non-autoregressive. If it is None than the value will be 1 if the model was trained with the
            glancing strategy, 10 otherwise (default=None).
        :param length_beam_size: the number of top lengths to consider for each sentence, akin to the beam size of
            the beam search (default=5).
        :return: tokenized translation of source sentence.
        """
        if iterations is None:
            iterations = 1 if self.glat_training else 10
        elif iterations < 1:
            raise ValueError("The number of iterations of mask-predict must be at least 1.")

        if length_beam_size < 1:
            raise ValueError("The number of lengths to consider for each sentence must be at least 1.")

        if not self._check_length_token(input_ids):
            raise ValueError(
                "You are not using the <length> token at the start of the source sentences, the model can not predict"
                "the target lengths."
            )

        if tgt_lang_token_id is None:
            raise ValueError("You should define the target language token id.")

        self.eval()
        device = next(self.parameters()).device
        batch_size = input_ids.shape[0]

        # Compute encodings
        e_mask = create_encoder_mask(input_ids, self.pad_token_id)
        encodings = self.encode(input_ids, e_mask)

        # Predict the best length_beam_size lengths for each sentence
        length_beams = self.predict_target_length(encodings, length_beam_size, e_mask)
        length_beams[length_beams < 2] = 2

        # Compute the largest length and the number of non-pad tokens
        max_length = length_beams.max().item()
        non_pad_tokens = max_length + 2

        # Build the length mask
        length_mask = torch.triu(input_ids.new(non_pad_tokens, non_pad_tokens).fill_(1).long(), 1)
        length_mask = torch.stack([length_mask[length_beams[batch] - 1] for batch in range(batch_size)], dim=0)

        # Initialize target tokens
        tgt_tokens = input_ids.new(batch_size, length_beam_size, non_pad_tokens).fill_(self.mask_token_id)
        tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * self.pad_token_id
        for i, lengths in enumerate(length_beams):
            lengths = lengths + torch.arange(0, length_beam_size * non_pad_tokens, non_pad_tokens).to(device)

            # Add eos token
            tgt_tokens[i].view(-1)[lengths] = self.eos_token_id

            # Add target language token if passed
            tgt_tokens[i].view(-1)[lengths + 1] = tgt_lang_token_id

        tgt_tokens = tgt_tokens.view(batch_size * length_beam_size, non_pad_tokens)

        # Duplicate encoder's output (without taking into account the encodings of the <length> token) and padding mask
        # to match the number of length beams
        start_token = 0 if self.length_token_id is None else 1
        duplicated_encodings = encodings[:, start_token:].repeat_interleave(length_beam_size, dim=0)
        duplicated_e_mask = e_mask[:, :, start_token:].repeat_interleave(length_beam_size, dim=0)

        # Mask-predict
        hyps, log_p, tokens_steps = self.__mask_predict(duplicated_encodings, duplicated_e_mask, tgt_tokens, iterations)

        # Reshape hypotheses and their log probabilities
        hyps = hyps.view(batch_size, length_beam_size, hyps.size(-1))
        log_p = log_p.view(batch_size, length_beam_size)

        # Compute the best lengths in terms of log probabilities
        tgt_lengths = (1 - length_mask).sum(-1)
        avg_log_prob = log_p / tgt_lengths.float()
        best_lengths = avg_log_prob.max(-1)[1]

        # Compute the best hypotheses
        output = torch.stack([hyps[batch, length, :] for batch, length in enumerate(best_lengths)], dim=0)

        # Compute the tokens at each step for the best hypothesis
        tokens_steps = tokens_steps.view(batch_size, length_beam_size, iterations + 1, non_pad_tokens)
        tokens_steps = torch.stack([tokens_steps[batch, length, :] for batch, length in enumerate(best_lengths)], dim=0)

        return output, tokens_steps
