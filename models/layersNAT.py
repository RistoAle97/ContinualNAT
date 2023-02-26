import torch
from torch import nn
from torch.functional import F
from models.transformer import positional_encoding


class DecoderLayerNAT(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 use_highway_layer: bool = True) -> None:
        super().__init__()
        # Parameters
        self.use_highway_layer = use_highway_layer

        # Highway linear layers
        if use_highway_layer:
            self.highway1 = nn.Linear(d_model, 1)
            self.highway2 = nn.Linear(d_model, 1)
            self.highway3 = nn.Linear(d_model, 1)
            self.highway4 = nn.Linear(d_model, 1)

        # Self-attention sublayer
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, layer_norm_eps)

        # Positional attention sublayer
        self.pos_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, layer_norm_eps)

        # Encoder-decoder attention sublayer
        self.enc_dec_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model, layer_norm_eps)

        # Feed-forward sublayer
        self.ff_linear1 = nn.Linear(d_model, dim_ff)
        self.dropout4 = nn.Dropout(dropout)
        self.ff_linear2 = nn.Linear(dim_ff, d_model)
        self.dropout5 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                e_mask: torch.Tensor = None,
                d_mask: torch.Tensor = None,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention sublayer
        self_output = self.self_attention(tgt_input, tgt_input, tgt_input, d_pad_mask, attn_mask=d_mask)
        if self.use_highway_layer:
            out_highway = F.sigmoid(self.highway1(self_output))
            sa_output = self_output * out_highway + (1 - out_highway) * self.dropout1(self_output)
        else:
            sa_output = self_output + self.dropout1(self_output)

        sa_output = self.norm1(sa_output)

        # Positional attention sublayer
        pos_output = positional_encoding(sa_output, self.d_model)
        pos_output = self.pos_attention(pos_output, pos_output, sa_output, d_pad_mask, attn_mask=d_mask)
        if self.use_highway_layer:
            out_highway = F.sigmoid(self.highway2(pos_output))
            pos_output = pos_output * out_highway + (1 - out_highway) * self.dropout2(pos_output)
        else:
            pos_output = pos_output + self.dropout2(pos_output)

        pos_output = self.norm2(pos_output)

        # Encoder-decoder attention sublayer
        enc_dec_output = self.enc_dec_attention(pos_output, src_input, src_input, e_pad_mask, attn_mask=e_mask)
        if self.use_highway_layer:
            out_highway = F.sigmoid(self.highway3(enc_dec_output))
            enc_dec_output = enc_dec_output * out_highway + (1 - out_highway) * self.dropout3(enc_dec_output)
        else:
            enc_dec_output = enc_dec_output + self.dropout3(enc_dec_output)

        enc_dec_output = self.norm3(enc_dec_output)

        # Feed-forward sublayer
        output = F.relu(self.ff_linear1(enc_dec_output))
        output = self.dropout4(output)
        output = self.ff_linear2(output)
        if self.use_highway_layer:
            out_highway = F.sigmoid(self.highway4(output))
            output = output * out_highway + (1 - out_highway) * self.dropout5(output)
        else:
            output = output + self.dropout5(output)

        output = self.norm4(output)
        return output


class DecoderNAT(nn.Module):

    def __init__(self,
                 decoder_layer: DecoderLayerNAT,
                 num_decoder_layers: int = 6) -> None:
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
        output = tgt_input
        for decoder_layer in self.layers:
            output = decoder_layer(src_input, output, e_mask, d_mask, e_pad_mask, d_pad_mask)

        return output
