import math

import torch
from torch import nn

from src.models.core.transformer_core import TransformerCore
from src.models.ftnat.config_ftnat import FTNATConfig
from src.modules.transformer_layers import TransformerDecoderLayer, TransformerDecoder
from src.modules.pooling import Fertility


class FTNAT(TransformerCore):

    def __init__(self, config: FTNATConfig) -> None:
        """
        The fertility NAT model (FT-NAT) by Gu et al. https://arxiv.org/pdf/1711.02281.pdf, the first
        non-autoregressive model for neural machine translation.
        """
        super().__init__(config)
        # Parameters
        self.max_fertilities = config.max_fertilities

        # Fertility
        self.fertility = Fertility(self.d_model, self.max_fertilities)

        # Decoder
        decoder_layer = TransformerDecoderLayer(self.d_model, self.n_heads, self.dim_ff, self.dropout, self.dropout_mha,
                                                self.dropout_ff, self.activation_ff, self.layer_norm_eps, True)
        norm = nn.LayerNorm(self.d_model, self.layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer, self.num_decoder_layers, norm)

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
