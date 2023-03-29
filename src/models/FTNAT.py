import torch
import math
from torch import nn
from . import TransformerCore
from ..modules import DecoderLayerNAT, DecoderNAT, Fertility


class FTNAT(TransformerCore):

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6,
                 max_fertilities: int = 50) -> None:
        """
        The fertility NAT model (FT-NAT) by Gu et al. (https://arxiv.org/pdf/1711.02281.pdf), the first
        non-autoregressive model for neural machine translation.
        :param vocab_size: shared vocabulary size.
        :param d_model: embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param num_encoder_layers: the number of encoder layers (default=6).
        :param num_decoder_layers: the number of decoder layers (default=6).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        :param max_fertilities: the maximum number of fertilities (default=50).
        """
        super().__init__(vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers,
                         dim_ff, dropout, layer_norm_eps)
        # Parameters
        self.max_fertilities = max_fertilities

        # Fertility
        self.fertility = Fertility(d_model, max_fertilities)

        # Decoder
        decoder_layer = DecoderLayerNAT(d_model, n_heads, dim_ff, dropout, layer_norm_eps, True)
        norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.decoder = DecoderNAT(decoder_layer, num_decoder_layers, norm)

    @staticmethod
    def copy_fertilities(src_input: torch.Tensor, fertilities: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = src_input.shape
        copied_embedding = torch.zeros([batch_size, seq_len, d_model])
        for i, fertility_batch in enumerate(fertilities):
            pos = 0
            for j, fertility in enumerate(fertility_batch):
                if fertility == 0:
                    continue

                copied_embedding[i, pos:pos+int(fertility), :] = src_input[i, j, :].repeat(1, int(fertility), 1)
                pos += int(fertility)

        return copied_embedding

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                d_mask: torch.Tensor = None,
                padding_mask: torch.Tensor = None,
                soft_copy: bool = False) -> torch.Tensor:
        """
        Process source sequence.
        """
        # Embeddings and positional encoding
        e_embeddings = self.src_embedding(src_input)
        e_input = self.positional_encoder(e_embeddings * math.sqrt(self.d_model))

        # Encoder and fertilities
        e_output = self.encoder(e_input, None, padding_mask)
        fertilities = self.fertility(e_output)
        if soft_copy:
            copied_embeddings = self.copy_fertilities(e_embeddings, fertilities)
        else:
            copied_embeddings = tgt_input

        # Decoder
        d_input = self.positional_encoder(copied_embeddings * math.sqrt(self.d_model))
        d_input = self.positional_dropout(d_input)
        d_output = self.decoder(d_input, e_output, d_mask, padding_mask, padding_mask)

        # Linear output
        output = self.linear_output(d_output)  # (batch_size, seq_len, tgt_vocab_size)
        return output

    def generate(self):
        pass
