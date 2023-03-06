import torch
from torch import nn
from torch.functional import F
from modules import DecoderLayerNAT, DecoderNAT
from . import TransformerCore


class RefineNAT(TransformerCore):

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int = None,
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 share_embeddings_src_trg: bool = True,
                 share_embeddings_trg_out: bool = True,
                 use_highway_layer: bool = True) -> None:
        super().__init__(src_vocab_size, tgt_vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers,
                         dim_ff, dropout, layer_norm_eps, share_embeddings_src_trg, share_embeddings_trg_out)
        # Parameters
        self.use_highway_layer = use_highway_layer

        # Decoders
        decoder_layer = DecoderLayerNAT(d_model, n_heads, dim_ff, dropout, layer_norm_eps, use_highway_layer)
        self.decoder = DecoderNAT(decoder_layer, num_decoder_layers)
        self.decoder1 = DecoderNAT(decoder_layer, num_decoder_layers)

        # Linear output
        self.linear_output1 = nn.Linear(d_model, self.tgt_vocab_size, bias=False)

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                d_mask: torch.Tensor = None,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Process masked source and target sequences.
        """
        # Embeddings and positional encoding
        src_input = self.src_embedding(src_input)  # (batch_size, seq_len, d_model)
        src_input = self.positional_encoder(src_input)
        tgt_input = self.tgt_embedding(tgt_input)  # (batch_size, seq_len, d_model)
        tgt_input = self.positional_encoder(tgt_input)

        # Encoder and decoder
        e_output = self.encoder(src_input, None, e_pad_mask)
        d_output = self.decoder(src_input, e_output, d_mask, None, d_pad_mask, e_pad_mask)
        d_output = self.decoder1(tgt_input, d_output, d_mask, None, d_pad_mask, e_pad_mask)

        # Linear output and softmax
        output = self.linear_output(d_output)  # (batch_size, seq_len, tgt_vocab_size)
        output = F.log_softmax(output, -1)
        return output

    def generate(self):
        pass
