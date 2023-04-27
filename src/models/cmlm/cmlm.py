import torch
from torch.functional import F
from src.models.core import TransformerCore
from src.models.cmlm import CMLMConfig
from src.modules import Pooler
from src.utils import init_bert_weights, create_masks
from typing import Tuple


class CMLM(TransformerCore):

    def __init__(self, config: CMLMConfig) -> None:
        """
        The Conditional Masked Language Model (CMLM) from Ghazvininejad et al. https://arxiv.org/pdf/1904.09324.pdf, a
        non-autoregressive model whose training is based on BERT by Devlin et al. https://arxiv.org/pdf/1810.04805.pdf
        and uses an iterative decoding strategy called "mask-predict" during inference.
        """
        super().__init__(config)
        # Token ids
        self.mask_token_id = config.mask_token_id
        self.length_token_id = config.length_token_id

        # Pooler layer after the encoder to predict the target sentence length
        self.pooler = Pooler(self.d_model)

        # Use BERT weight initialization
        self.apply(init_bert_weights)

        # Train and validation losses
        self.train_metrics["cmlm_logits_loss"] = 0
        self.train_metrics["cmlm_lengths_loss"] = 0

    def __check_length_token(self, input_ids: torch.Tensor) -> bool:
        is_using_length_token = (input_ids[:, 0] == self.length_token_id)
        return is_using_length_token.all()

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                e_mask: torch.Tensor = None,
                d_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process source and target sequences.
        """
        if not self.__check_length_token(src_input):
            raise ValueError("The token <length> is not used by one or more tokenized sentence, the model needs such"
                             "token to predict the target lengths.")

        # Embeddings and positional encoding
        src_embeddings = self.embedding(src_input)  # (bsz, seq_len, d_model)
        tgt_embeddings = self.embedding(tgt_input)  # (bsz, seq_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)
        tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)

        # Encoder and decoder
        e_output = self.encoder(src_embeddings, e_mask)
        predicted_lengths = self.pooler(e_output)  # (bsz, seq_len)
        d_output = self.decoder(tgt_embeddings, e_output[:, 1:], d_mask, e_mask[:, :, 1:])

        # Linear output
        output = self.linear_output(d_output)  # (bsz, seq_len, vocab_size)
        return output, predicted_lengths

    def predict_target_length(self, e_output: torch.Tensor, n_lengths: int = 1) -> torch.Tensor:
        """
        Computes the encodings of the target sentence length given the encoder's output.
        :param e_output: the encoder's output.
        :param n_lengths: the number of possible lengths to consider for each sentence.
        :return: the encodings of the target sentence length.
        """
        length_logits = self.pooler(e_output)
        length_logits = F.log_softmax(length_logits, dim=-1)
        lengths = length_logits.topk(n_lengths, dim=-1)[1]
        return lengths

    def compute_loss(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor,
                     predicted_lengths: torch.Tensor,
                     target_lengths: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # Logits loss
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        logits_loss = F.cross_entropy(logits, labels, ignore_index=self.pad_token_id)

        # Length loss
        predicted_lengths = predicted_lengths.contiguous().view(-1, predicted_lengths.size(-1))
        target_lengths = target_lengths.contiguous().view(-1)
        lengths_loss = F.cross_entropy(predicted_lengths, target_lengths)

        # Combine the losses
        loss = logits_loss + lengths_loss
        return loss, logits_loss.item(), lengths_loss.item()

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        decoder_input_ids = batch["decoder_input_ids"]
        target_lengths = batch["target_lengths"]

        # Create masks
        e_mask, d_mask = create_masks(input_ids, decoder_input_ids, self.pad_token_id, None)

        # Compute loss
        logits, predicted_lengths = self(input_ids, decoder_input_ids, e_mask=e_mask, d_mask=d_mask)
        loss, logits_loss, lengths_loss = self.compute_loss(logits, labels, predicted_lengths, target_lengths)

        # Update metrics for logging
        self.train_metrics["train_loss"] += loss.item()
        self.train_metrics["cmlm_logits_loss"] += logits_loss
        self.train_metrics["cmlm_lengths_loss"] += lengths_loss
        return loss

    def __mask_predict(self,
                       encodings: torch.Tensor,
                       e_mask: torch.Tensor,
                       tgt_input: torch.Tensor,
                       iterations: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Parameters
            batch_size, seq_len = tgt_input.size()
            device = next(self.parameters()).device

            # Make first prediction in a fully non-autoregressive way
            d_mask = tgt_input.ne(self.pad_token_id).unsqueeze(1).to(device)  # this mask will never change
            tgt_lengths = seq_len - tgt_input.ne(self.mask_token_id).sum(dim=-1)
            output = self.decode(tgt_input, encodings, d_mask=d_mask, e_mask=e_mask)
            logits = F.log_softmax(output, dim=-1)
            p_tokens, tokens = logits.max(dim=-1)  # tokens probabilites and their ids
            tokens.view(-1)[(d_mask == 0).view(-1).nonzero()] = self.pad_token_id
            p_tokens.view(-1)[(d_mask == 0).view(-1).nonzero()] = 0

            # Mask-predict iterations
            for i in range(1, iterations):
                # Compute the number of masks per sentence
                n_masks = (tgt_lengths * (1.0 - i / iterations)).int()

                # Compute the indexes of the worst tokens in terms of probability
                masks = [p_tokens[batch, :].topk(max(1, n_masks[batch]), largest=False, sorted=False)[1] for batch in
                         range(batch_size)]
                masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
                masks = torch.stack(masks, dim=0)

                # Apply the masks
                masks = masks + torch.arange(0, batch_size * seq_len, seq_len, device=device).unsqueeze(1)
                tokens.view(-1)[masks.view(-1)] = self.mask_token_id

                # Compute the new tokens and their probabilities
                output = self.decode(tokens, encodings, d_mask=d_mask, e_mask=e_mask)
                logits = F.log_softmax(output, dim=-1)
                new_p_tokens, new_tokens = logits.max(dim=-1)

                # Update the output tokens and probabilities
                p_tokens.view(-1)[masks.view(-1)] = new_p_tokens.view(-1)[masks.view(-1)]
                p_tokens.view(-1)[(d_mask == 0).view(-1).nonzero()] = 0
                tokens.view(-1)[masks.view(-1)] = new_tokens.view(-1)[masks.view(-1)]
                tokens.view(-1)[(d_mask == 0).view(-1).nonzero()] = self.pad_token_id

            # Sum the log probabilities of the tokens for each sentence
            log_p_tokens = p_tokens.sum(-1)
            return tokens, log_p_tokens

    def generate(self,
                 input_ids: torch.Tensor,
                 tgt_lang_token_id: int = None,
                 iterations: int = 10,
                 length_beam_size: int = 5) -> torch.Tensor:
        """
        Generate tokens during inference by using the mask-predict algorithm by Ghazvininejad et al.
        https://arxiv.org/pdf/1904.09324.pdf.
        :param input_ids: the tokenized source sentence.
        :param tgt_lang_token_id: the target language token id, if none is passed, then no token will appended ad the
            end of the target tokens (default=None).
        :param iterations: the number of iterations of the mask-predict. If its value is equal to 1, then the decoding
            will be purely non-autoregressive (default=10).
        :param length_beam_size: the number of top lengths to consider for each sentence, akin to the beam size of
            the beam search (default=5).
        :return: tokenized translation of source sentence.
        """
        if iterations < 1:
            raise ValueError("The number of iterations of mask-predict must be at least 1.")

        if length_beam_size < 1:
            raise ValueError("The number of lengths to consider for each sentence must be at least 1.")

        if not self.__check_length_token(input_ids):
            raise ValueError("You are not using the <length> token at the start of the source sentences,"
                             "the model can not predict the target lengths.")

        self.eval()
        device = next(self.parameters()).device
        batch_size = input_ids.shape[0]

        # Compute encodings
        e_mask = input_ids.ne(self.pad_token_id).unsqueeze(1).to(device)
        encodings = self.encode(input_ids, e_mask)

        # Predict the best length_beam_size lengths for each sentence
        length_beams = self.predict_target_length(encodings, length_beam_size)
        length_beams[length_beams < 2] = 2

        # Compute the largest length and the number of non-pad tokens
        max_length = length_beams.max().item()
        non_pad_tokens = max_length + 2 if tgt_lang_token_id is not None else max_length + 1

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

        # Duplicate encoder's output (without taking into account the encodings of the <length> token) and padding mask
        # to match the number of length beams
        duplicated_encodings = encodings[:, 1:].repeat_interleave(length_beam_size, dim=0)
        duplicated_e_mask = e_mask[:, :, 1:].repeat_interleave(length_beam_size, dim=0)

        # Mask-predict
        hypotheses, log_probabilities = self.__mask_predict(duplicated_encodings, duplicated_e_mask,
                                                            tgt_tokens, iterations)

        # Reshape hypotheses and their log probabilities
        hypotheses = hypotheses.view(batch_size, length_beam_size, hypotheses.size(-1))
        log_probabilities = log_probabilities.view(batch_size, length_beam_size)

        # Compute the best lengths in terms of log probabilities
        tgt_lengths = (1 - length_mask).sum(-1)
        avg_log_prob = log_probabilities / tgt_lengths.float()
        best_lengths = avg_log_prob.max(-1)[1]

        # Retrieve best hypotheses
        output = torch.stack([hypotheses[batch, length, :] for batch, length in enumerate(best_lengths)], dim=0)
        return output
