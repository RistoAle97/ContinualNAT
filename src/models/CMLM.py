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
        predicted_length = self.pooler(e_output, True).unsqueeze(1)
        return predicted_length

    def __mask_predict(self,
                       input_ids: torch.Tensor,
                       tgt_lang_token_id: int,
                       pad_token_id: int,
                       mask_token_id: int,
                       iterations: int = 10) -> torch.Tensor:
        with torch.no_grad():
            # Parameters
            batch_size, seq_len = input_ids.size()
            device = next(self.parameters()).device

            # Encode the input tokens
            e_pad_mask = (input_ids == pad_token_id).to(device)
            encodings = self.encode(input_ids, e_pad_mask=e_pad_mask)

            # Predict the target lengths and take the largest one
            target_lengths = self.predict_target_length(encodings)
            max_predicted_length = target_lengths.max()

            # Initialize outputs <mask0>...<maskT-1> <tgt_lang_code_id> <pad>...<pad> where T is the predicted length
            output = torch.ones(batch_size, max_predicted_length.item() + 1, dtype=torch.int,
                                device=device).fill_(pad_token_id)
            for i, length in enumerate(target_lengths):
                output[i, :length] = mask_token_id
                output[i, length] = tgt_lang_token_id

            # Make first prediction in a fully non-autoregressive way
            d_pad_mask = (output == pad_token_id)  # this mask will never change
            output = self.decode(encodings, output, e_pad_mask=e_pad_mask, d_pad_mask=d_pad_mask)
            logits = F.log_softmax(output)
            p_tokens, tokens = logits.max(dim=-1)  # tokens probabilites and their ids
            tokens.view(-1)[d_pad_mask.view(-1).nonzero()] = pad_token_id
            p_tokens.view(-1)[d_pad_mask.view(-1).nonzero()] = 1.0

            # Mask-predict iterations
            for i in range(1, iterations):
                # Compute the number of masks per sentence
                n_masks = (target_lengths * (1.0 - i / iterations)).int()

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

            return tokens

    def generate(self,
                 input_ids: torch.Tensor,
                 sos_token_id: int,
                 pad_token_id: int,
                 mask_token_id: int,
                 iterations: int = 10) -> torch.Tensor:
        return self.__mask_predict(input_ids, sos_token_id, pad_token_id, mask_token_id, iterations)
