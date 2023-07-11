import torch
import copy
from torch import nn
from src.modules.transformer_layers import MultiHeadAttention, FeedForwardLayer
from src.modules.connections import ResidualConnection, HighwayConnection
from src.modules.positional_encoding import PositionalEncoding


class DecoderLayerNAT(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 dropout_mha: float = 0.0,
                 dropout_ff: float = 0.0,
                 activation_ff: str = "relu",
                 layer_norm_eps: float = 1e-6,
                 use_highway_layer: bool = False) -> None:
        """
        The non-autoregressive transformer decoder layer as first introduced by Gu et al.
        https://arxiv.org/pdf/1711.02281.pdf. Its structure is the same as the transformer base from Vaswani et al.
        https://arxiv.org/pdf/1706.03762.pdf with an additional layer (called positional attention) placed between the
        self-attention and the encoder-decoder attention layers. The positional attention layer expects the positional
        encoding of the self-attention output as its query and key, while expecting the output of the
        self-attention layer as its value. Differently from the original implementation, we apply pre-norm instead of
        post-norm.
        :param d_model: the model's embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        :param use_highway_layer: whether to use a highway connection around each sublayer, if set to False then
            residual connections will be used (default=False)
        """
        super().__init__()
        # Parameters
        self.use_highway_layer = use_highway_layer
        self.positional_encoder = PositionalEncoding(d_model, dropout=0)

        # Connections around each layer
        if use_highway_layer:
            self.block_connections = nn.ModuleList([HighwayConnection(d_model, dropout) for _ in range(4)])
        else:
            self.block_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(4)])

        # Self-attention sublayer
        self.self_att = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.sa_norm = nn.LayerNorm(d_model, layer_norm_eps)

        # Positional attention sublayer
        self.pos_att = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.pos_norm = nn.LayerNorm(d_model, layer_norm_eps)

        # Encoder-decoder attention sublayer
        self.mha = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.mha_norm = nn.LayerNorm(d_model, layer_norm_eps)

        # Feed-forward sublayer
        self.ff_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ff = FeedForwardLayer(d_model, dim_ff, dropout_ff, activation_ff)
        self.ff_dropout = nn.Dropout(dropout)

    '''def _maybe_layer_norm(self,
                          x: torch.Tensor,
                          norm: nn.Module,
                          before: bool = False,
                          after: bool = False) -> torch.Tensor:
        assert before ^ after
        if after ^ self.norm_first:
            return norm(x)
        else:
            return x'''

    def forward(self,
                tgt_input: torch.Tensor,
                e_output: torch.Tensor,
                d_mask: torch.Tensor = None,
                e_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Process masked source and target sequences.
        """
        # Self-attention sublayer
        sa_out = self.sa_norm(tgt_input)
        sa_output = self.self_att(sa_out, sa_out, sa_out, d_mask)
        sa_output = self.block_connections[0](tgt_input, sa_output)
        # sa_output = self._maybe_layer_norm(sa_output, self.norm1, after=True)

        # Positional attention sublayer
        pos_out = self.positional_encoder(sa_output)
        pos_out = self.pos_norm(pos_out)
        pos_out = self.pos_attention(pos_out, pos_out, sa_output, d_mask)
        pos_out = self.block_connections[1](sa_output, pos_out)
        # pos_output = self._maybe_layer_norm(pos_output, self.norm2, after=True)

        # Encoder-decoder attention sublayer
        mha_out = self.pos_norm(pos_out)
        mha_out = self.encdec_attention(mha_out, e_output, e_output, e_mask)
        mha_out = self.block_connections[2](pos_out, mha_out)
        # encdec_output = self._maybe_layer_norm(encdec_output, self.norm3, after=True)

        # Feed-forward sublayer
        ff_out = self.ff_norm(mha_out)
        ff_out = self.ff(ff_out)
        out = self.block_connections[3](mha_out, ff_out)
        # out = self._maybe_layer_norm(output, self.norm4, after=True)
        return out


class DecoderNAT(nn.Module):

    def __init__(self, decoder_layer: DecoderLayerNAT, num_decoder_layers: int = 6, norm: nn.Module = None) -> None:
        """
        The non-autoregressive transformer decoder by Gu et al. https://arxiv.org/pdf/1711.02281.pdf.
        :param decoder_layer: the non-autoregressive decoder layer.
        :param num_decoder_layers: the number of decoder layers (default=6).
        """
        super().__init__()
        # Parameters
        self.num_layers = num_decoder_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        self.norm = norm

    def forward(self,
                e_output: torch.Tensor,
                tgt_input: torch.Tensor,
                d_mask: torch.Tensor = None,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Process masked source and target sequences.
        """
        output = tgt_input
        for decoder_layer in self.layers:
            output = decoder_layer(e_output, output, d_mask, e_pad_mask, d_pad_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
