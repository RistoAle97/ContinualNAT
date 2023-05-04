import torch
import math
from torch import nn
from src.models.core import TransformerCore
from src.models.refinenat import RefineNATConfig
from src.modules import DecoderLayerNAT, DecoderNAT


class RefineNAT(TransformerCore):

    def __init__(self, config: RefineNATConfig) -> None:
        """
        The RefineNAT model by Lee et al. (https://arxiv.org/abs/1802.06901), the first NAT model whose decoding is
        based on iterative refinement.
        """
        super().__init__(config)
        # Parameters
        self.use_highway_layer = config.use_highway_layer

        # Decoders
        decoder_layer = DecoderLayerNAT(self.d_model, self.n_heads, self.dim_ff, self.dropout, self.layer_norm_eps,
                                        True, self.use_highway_layer)
        norm = nn.LayerNorm(self.d_model, self.layer_norm_eps)
        self.decoder = DecoderNAT(decoder_layer, self.num_decoder_layers, norm)
        norm1 = nn.LayerNorm(self.d_model, self.layer_norm_eps)
        self.decoder1 = DecoderNAT(decoder_layer, self.num_decoder_layers, norm1)

        # Linear output
        self.linear_output1 = nn.Linear(self.d_model, self.tgt_vocab_size, bias=False)

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                d_mask: torch.Tensor = None,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Process masked source and target sequences.
        """
        # Encoder
        e_input = self.src_embedding(src_input)  # (batch_size, seq_len, d_model)
        e_input = self.positional_encoder(e_input * math.sqrt(self.d_model))
        e_output = self.encoder(e_input, None, e_pad_mask)

        # Decoder
        d_input = self.tgt_embedding(tgt_input)  # (batch_size, seq_len, d_model)
        d_input = self.positional_encoder(d_input * math.sqrt(self.d_model))
        d_output = self.decoder(src_input, e_output, d_mask, None, d_pad_mask, e_pad_mask)
        d_output = self.decoder1(d_input, d_output, d_mask, None, d_pad_mask, e_pad_mask)

        # Linear output
        output = self.linear_output(d_output)  # (batch_size, seq_len, tgt_vocab_size)
        return output

    def generate(self):
        pass
