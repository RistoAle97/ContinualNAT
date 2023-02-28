import torch
from torch import nn
from torch.functional import F
from models.transformer import positional_encoding
from modules.connections import ResidualConnection, HighwayConnection


class DecoderLayerNAT(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 use_highway_layer: bool = True) -> None:
        """
        The non-autoregressive transformer decoder layer as first introduced by Gu et al.
        https://arxiv.org/pdf/1711.02281.pdf. Its structure is the same as the transformer base from Vaswani et al.
        https://arxiv.org/pdf/1706.03762.pdf with an additional layer (called positional attention) placed between the
        self-attention and the encoder-decoder attention layers. The positional attention layer expects the positional
        encoding of the self-attention output as its query and key, while expecting the output of the
        self-attention layer as its value.
        :param d_model: the model's embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-5).
        :param use_highway_layer: whether to use a highway connection around each sublayer, if set to False then
            residual connections will be used (default=True)
        """
        super().__init__()
        # Parameters
        self.use_highway_layer = use_highway_layer

        # Connections around each layer
        if use_highway_layer:
            self.block_connections = nn.ModuleList([HighwayConnection(d_model, dropout) for _ in range(4)])
        else:
            self.block_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(4)])

        # Self-attention sublayer
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model, layer_norm_eps)

        # Positional attention sublayer
        self.pos_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, layer_norm_eps)

        # Encoder-decoder attention sublayer
        self.encdec_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model, layer_norm_eps)

        # Feed-forward sublayer
        self.ff_linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.ff_linear2 = nn.Linear(dim_ff, d_model)
        self.norm4 = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                e_mask: torch.Tensor = None,
                d_mask: torch.Tensor = None,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Process masked source and target sequences.
        """
        # Self-attention sublayer
        sa_output = self.self_attention(tgt_input, tgt_input, tgt_input, d_pad_mask, attn_mask=d_mask)[0]
        sa_output = self.block_connections[0](tgt_input, sa_output)
        sa_output = self.norm1(sa_output)

        # Positional attention sublayer
        pos_output = positional_encoding(sa_output, self.d_model)
        pos_output = self.pos_attention(pos_output, pos_output, sa_output, d_pad_mask, attn_mask=d_mask)[0]
        pos_output = self.block_connections[1](sa_output, pos_output)
        pos_output = self.norm2(pos_output)

        # Encoder-decoder attention sublayer
        encdec_output = self.encdec_attention.forward(pos_output, src_input, src_input, e_pad_mask, attn_mask=e_mask)[0]
        encdec_output = self.block_connections[2](pos_output, encdec_output)
        encdec_output = self.norm3(encdec_output)

        # Feed-forward sublayer
        output = F.relu(self.ff_linear1(encdec_output))
        output = self.dropout4(output)
        output = self.ff_linear2(output)
        output = self.block_connections[3](encdec_output, output)
        output = self.norm4(output)
        return output


class DecoderNAT(nn.Module):

    def __init__(self,
                 decoder_layer: DecoderLayerNAT,
                 num_decoder_layers: int = 6) -> None:
        """
        The non-autoregressive transformer decoder by Gu et al. https://arxiv.org/pdf/1711.02281.pdf.
        :param decoder_layer: the non-autoregressive decoder layer.
        :param num_decoder_layers: the number of decoder layers (default=6).
        """
        super().__init__()
        # Parameters
        self.num_layers = num_decoder_layers
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                e_mask: torch.Tensor = None,
                d_mask: torch.Tensor = None,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Process masked source and target sequences.
        """
        output = tgt_input
        for decoder_layer in self.layers:
            output = decoder_layer(src_input, output, e_mask, d_mask, e_pad_mask, d_pad_mask)

        return output
