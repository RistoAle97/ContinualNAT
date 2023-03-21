import torch
import math
from torch import nn
from torch.functional import F
from . import TransformerCore
from ..modules import Pooler
from typing import Tuple


class CMLM(TransformerCore):

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int = None,
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6,
                 norm_first: bool = False,
                 share_embeddings_src_tgt: bool = True,
                 share_embeddings_tgt_out: bool = True) -> None:
        """
        The Conditional Masked Language Model from Ghazvininejad et al. https://arxiv.org/pdf/1904.09324.pdf, a
        non-autoregressive model whose training is based on BERT by Devlin et al. https://arxiv.org/pdf/1810.04805.pdf
        and uses an iterative decoding strategy called mask-predict during inference.
        :param src_vocab_size: input language vocabulary size.
        :param tgt_vocab_size: target language vocabulary size, if no value is passed, then it will have the same size
            of the source one.
        :param d_model: embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param num_encoder_layers: the number of encoder layers (default=6).
        :param num_decoder_layers: the number of decoder layers (default=6).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        :param norm_first: if True, encoder and decoder layers will perform LayerNorms before other attention and
            feedforward operations, otherwise after. Default: False (after).
        :param share_embeddings_src_tgt: whether to share the weights beetween source and target embedding layers
            (default=True).
        :param share_embeddings_tgt_out: whether to share the weights beetween the target embeddings and the linear
            output (default=True).
        """
        super().__init__(src_vocab_size, tgt_vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers,
                         dim_ff, dropout, layer_norm_eps, norm_first, share_embeddings_src_tgt,
                         share_embeddings_tgt_out)
        # Pooler layer after the encoder to predict the target sentence length
        self.pooler = Pooler(d_model)

        # Use BERT weight initialization
        self.apply(self._init_bert_weights)

    @staticmethod
    def _init_bert_weights(module: nn.Module) -> None:
        """
        Initialize module's weights following BERT https://arxiv.org/pdf/1810.04805.pdf.
        :param module: the module to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process source and target sequences.
        """
        # Embeddings and positional encoding
        e_input = self.src_embedding(src_input)  # (batch_size, seq_len, d_model)
        d_input = self.tgt_embedding(tgt_input)  # (batch_size, seq_len, d_model)
        e_input = self.positional_encoder(e_input * math.sqrt(self.d_model))
        d_input = self.positional_encoder(d_input * math.sqrt(self.d_model))

        # Encoder and decoder
        e_output = self.encoder(e_input, None, e_pad_mask)
        predicted_length = self.pooler(e_output)  # (batch_size, seq_len)
        d_output = self.decoder(d_input, e_output, None, None, d_pad_mask, e_pad_mask)

        # Linear output
        output = self.linear_output(d_output)  # (batch_size, seq_len, tgt_vocab_size)
        return output, predicted_length

    def predict_target_length(self, e_output: torch.Tensor) -> torch.Tensor:
        """
        Computes the encodings of the target sentence length given the encoder's output.
        :param e_output: the encoder's output.
        :return: the encodings of the target sentence length.
        """
        predicted_length = self.pooler(e_output, True).unsqueeze(1)
        return predicted_length

    def __mask_predict(self,
                       encodings: torch.Tensor,
                       e_pad_mask: torch.tensor,
                       tgt_input: torch.Tensor,
                       pad_token_id: int,
                       mask_token_id: int,
                       iterations: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Parameters
            batch_size, seq_len = tgt_input.size()
            device = next(self.parameters()).device

            # Make first prediction in a fully non-autoregressive way
            d_pad_mask = (tgt_input == pad_token_id).to(device)  # this mask will never change
            tgt_lengths = seq_len - d_pad_mask.sum(dim=1)
            output = self.decode(encodings, tgt_input, e_pad_mask=e_pad_mask, d_pad_mask=d_pad_mask)
            logits = F.log_softmax(output)
            p_tokens, tokens = logits.max(dim=-1)  # tokens probabilites and their ids
            tokens.view(-1)[d_pad_mask.view(-1).nonzero()] = pad_token_id
            p_tokens.view(-1)[d_pad_mask.view(-1).nonzero()] = 1.0

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
                tokens.view(-1)[masks.view(-1)] = mask_token_id

                # Compute the new tokens and their probabilities
                output = self.decode(encodings, tokens, e_pad_mask=e_pad_mask, d_pad_mask=d_pad_mask)
                logits = F.log_softmax(output, dim=-1)
                new_p_tokens, new_tokens = logits.max(dim=-1)

                # Update the output tokens and probabilities
                p_tokens.view(-1)[masks.view(-1)] = new_p_tokens.view(-1)[masks.view(-1)]
                p_tokens.view(-1)[d_pad_mask.view(-1).nonzero()] = 1.0
                tokens.view(-1)[masks.view(-1)] = new_tokens.view(-1)[masks.view(-1)]
                tokens.view(-1)[d_pad_mask.view(-1).nonzero()] = pad_token_id

            # Sum the log probabilities of the tokens for each sentence
            log_p_tokens = p_tokens.log().sum(-1)
            return tokens, log_p_tokens

    def generate(self,
                 input_ids: torch.Tensor,
                 pad_token_id: int,
                 mask_token_id: int,
                 tgt_lang_token_id: int = None,
                 iterations: int = 10,
                 length_beam_size: int = 5) -> torch.Tensor:
        """
        Generate tokens during inference by using the mask-predict algorithm by Ghazvininejad et al.
        https://arxiv.org/pdf/1904.09324.pdf.
        :param input_ids: the tokenized source sentence.
        :param pad_token_id: the pad token id.
        :param mask_token_id: the mask token id.
        :param tgt_lang_token_id: the target language token id, if none is passed, then no token will appended ad the
            end of the target tokens (default=None).
        :param iterations: the number of iterations of the mask-predict (default=10).
        :param length_beam_size: the number of top lengths to consider for each sentence, akin to the beam size of
            the beam search (default=5).
        :return: tokenized translation of source sentence.
        """
        batch_size = input_ids.shape[0]

        # Compute encodings
        e_pad_mask = (input_ids == pad_token_id).to(input_ids.device)
        encodings = self.encode(input_ids, e_pad_mask=e_pad_mask)

        # Predict the best length_beam_size lengths for each sentence
        target_lengths = self.predict_target_length(encodings)
        length_beams = target_lengths.topk(length_beam_size, dim=1)[1]
        length_beams[length_beams < 2] = 2

        # Compute the largest length and the number of non-pad tokens
        max_length = length_beams.max().item()
        non_pad_tokens = max_length + 1 if tgt_lang_token_id is not None else max_length

        # Build the length mask
        length_mask = torch.triu(input_ids.new(non_pad_tokens, non_pad_tokens).fill_(1).long(), 1)
        length_mask = torch.stack([length_mask[length_beams[batch] - 1] for batch in range(batch_size)], dim=0)

        # Initialize target tokens
        tgt_tokens = input_ids.new(batch_size, length_beam_size, non_pad_tokens).fill_(mask_token_id)
        if tgt_lang_token_id is not None:
            for i, lengths in enumerate(length_beams):
                lengths = lengths + torch.arange(0, length_beam_size * non_pad_tokens, non_pad_tokens)
                tgt_tokens[i].view(-1)[lengths] = tgt_lang_token_id

        tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * pad_token_id
        tgt_tokens = tgt_tokens.view(batch_size * length_beam_size, non_pad_tokens)

        # Duplicate encoder's output and padding mask to match the number of length beams
        duplicated_encodings = encodings.unsqueeze(2).repeat(1, 1, length_beam_size, 1)\
            .view(-1, batch_size * length_beam_size, encodings.size(-1))
        duplicated_e_pad_mask = e_pad_mask.unsqueeze(1).repeat(1, length_beam_size, 1)\
            .view(batch_size * length_beam_size, -1)

        # Mask-predict
        hypotheses, log_probabilities = self.__mask_predict(duplicated_encodings, duplicated_e_pad_mask, tgt_tokens,
                                                            pad_token_id, mask_token_id, iterations)

        # Reshape hypotheses and their log probabilities
        hypotheses = hypotheses.view(batch_size, length_beam_size, max_length)
        log_probabilities = log_probabilities.view(batch_size, length_beam_size, max_length)

        # Compute the best lengths in terms of log probabilities
        tgt_lengths = (1 - length_mask).sum(-1)
        avg_log_prob = log_probabilities / tgt_lengths.float()
        best_lengths = avg_log_prob.max(-1)[1]

        # Retrieve best hypotheses
        output = torch.stack([hypotheses[batch, length, :] for batch, length in enumerate(best_lengths)], dim=0)
        return output
