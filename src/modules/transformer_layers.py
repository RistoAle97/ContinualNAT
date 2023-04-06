import torch
import copy
from torch import nn
from torch.functional import F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0

        # Parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for query, key and value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection and dropout
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def self_attention(self,
                       query: torch.Tensor,
                       key: torch.Tensor,
                       value: torch.Tensor,
                       mask: torch.Tensor = None) -> torch.Tensor:
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores /= self.d_model ** 0.5
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        p_attn = attn_scores.softmax(dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)

        # Perform linear operation and split into n_heads heads
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim)  # (bsz, seq_len, n_heads, head_dim)
        q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.head_dim)  # (bsz, seq_len, n_heads, head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim)  # (bsz, seq_len, n_heads, head_dim)

        # Transpose to get shapes (bsz, n_heads, seq_len, d_model)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute self-attention
        scores = self.self_attention(q, k, v, mask)  # (bsz, n_heads, seq_len, d_model)

        # Concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(concat)  # (bsz, seq_len, d_model)
        return output


class FeedForwardLayer(nn.Module):

    def __init__(self, d_model: int = 512, dim_ff: int = 2048, dropout: float = 0.0, activation: str = "relu") -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        if activation not in ["relu", "gelu"]:
            raise ValueError("The activation function of the feed-forward sublayer must be relu or gelu.")

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 dropout_mha: float = 0.0,
                 dropout_ff: float = 0.0,
                 layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        # Self-attention sublayer
        self.mha_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.mha_dropout = nn.Dropout(dropout)

        # Feed-forward sublayer
        self.ff_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ff = FeedForwardLayer(d_model, dim_ff, dropout_ff)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self,
                src_input: torch.Tensor,
                src_mask: torch.Tensor = None) -> torch.Tensor:
        mha_out = self.mha_norm(src_input)
        mha_out = self.mha.forward(mha_out, mha_out, mha_out, src_mask)
        mha_out = src_input + self.mha_dropout(mha_out)
        ff_out = self.ff_norm(mha_out)
        ff_out = self.ff(ff_out)
        out = mha_out + self.ff_dropout(ff_out)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int = 6, norm: nn.LayerNorm = None) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self,
                src_input: torch.Tensor,
                src_mask: torch.Tensor = None) -> torch.Tensor:
        output = src_input
        for encoder_layer in self.layers:
            output = encoder_layer(output, src_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 dropout_mha: float = 0.0,
                 dropout_ff: float = 0.0,
                 layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        # Self-attention sublayer
        self.sa_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.sa = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.sa_dropout = nn.Dropout(dropout)

        # Encoder-decoder attention sublayer
        self.mha_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.mha_dropout = nn.Dropout(dropout)

        # Feed-forward sublayer
        self.ff_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ff = FeedForwardLayer(d_model, dim_ff, dropout)
        self.ff_dropout = nn.Dropout(dropout_ff)

    def forward(self,
                tgt_input: torch.Tensor,
                e_output: torch.Tensor,
                d_mask: torch.Tensor = None,
                e_mask: torch.Tensor = None) -> torch.Tensor:
        sa_out = self.sa_norm(tgt_input)
        sa_out = self.sa(sa_out, sa_out, sa_out, d_mask)
        sa_out = tgt_input + self.sa_dropout(sa_out)
        mha_out = self.mha_norm(sa_out)
        mha_out = self.mha(mha_out, e_output, e_output, e_mask)
        mha_out = sa_out + self.mha_dropout(mha_out)
        ff_out = self.ff_norm(mha_out)
        ff_out = self.ff(ff_out)
        out = mha_out + self.ff_dropout(ff_out)
        return out


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int = 6, norm: nn.LayerNorm = None):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self,
                tgt_input: torch.Tensor,
                e_output: torch.Tensor,
                d_mask: torch.Tensor = None,
                e_mask: torch.Tensor = None) -> torch.Tensor:
        output = tgt_input
        for encoder_layer in self.layers:
            output = encoder_layer(output, e_output, d_mask, e_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output