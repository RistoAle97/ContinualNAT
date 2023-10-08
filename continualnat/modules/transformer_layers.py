import copy

import torch
from torch import nn
from torch.functional import F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.0) -> None:
        """
        The multi-head attention sublayer from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf).
        :param d_model: the model's embedding dimension (default=512).
        :param n_heads: the number of heads (default=8).
        :param dropout: the dropout value (default= 0.0).
        """
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

    def __self_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores /= self.d_model**0.5
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        p_attn = attn_scores.softmax(dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
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
        scores = self.__self_attention(q, k, v, mask)  # (bsz, n_heads, seq_len, d_model)

        # Concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(concat)  # (bsz, seq_len, d_model)
        return output


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int = 512, dim_ff: int = 2048, dropout: float = 0.0, activation: str = "relu") -> None:
        """
        The feed-forward sublayer from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf).
        :param d_model: the model's embedding dimension (default=512).
        :param dim_ff: size of the intermediate linear transformation (default=2048).
        :param dropout: the dropout value (default=0.0).
        :param activation: the activation function, can be either ReLU or GeLu (default="relu").
        """
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
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        dropout_mha: float = 0.0,
        dropout_ff: float = 0.0,
        activation_ff: str = "relu",
        layer_norm_eps: float = 1e-6,
    ) -> None:
        """
        The transformer encoder layer from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf). The layer
        is made up of one multi-head attention sublayer followed by a feed-forward sublayer. Differently from the paper,
        this implementation uses pre-norm in the residual connection.
        :param d_model: the model's embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param dropout_mha: the dropout value for the multi-head attention (default=0.0).
        :param dropout_ff: the dropout value for the feed-forward sublayer (default=0.0).
        :param activation_ff: the activation function for the feed-forward sub-layer, can be either ReLU or GeLU
            (default="relu").
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        """
        super().__init__()
        # Multi-head attention sublayer
        self.mha_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.mha_dropout = nn.Dropout(dropout)

        # Feed-forward sublayer
        self.ff_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ff = FeedForwardLayer(d_model, dim_ff, dropout_ff, activation_ff)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, src_embeddings: torch.Tensor, e_mask: torch.Tensor = None) -> torch.Tensor:
        # Multi-head attention sublayer
        mha_out = self.mha_norm(src_embeddings)
        mha_out = self.mha(mha_out, mha_out, mha_out, e_mask)
        mha_out = src_embeddings + self.mha_dropout(mha_out)

        # Feed-forward sublayer
        ff_out = self.ff_norm(mha_out)
        ff_out = self.ff(ff_out)
        out = mha_out + self.ff_dropout(ff_out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int = 6, norm: nn.LayerNorm = None) -> None:
        """
        The encoder from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf). Following the actual
        implementation of the paper, a LayerNorm layer is put at the end of the encoder layers stack.
        :param encoder_layer: transformer's encoder layer that will be used in order to build the stack of
            encoder layers.
        :param num_layers: the number of layers (default=6).
        :param norm: the layer normalization that should be at the end of the encoder layers stack (default=None).
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src_embeddings: torch.Tensor, e_mask: torch.Tensor = None) -> torch.Tensor:
        output = src_embeddings
        for encoder_layer in self.layers:
            output = encoder_layer(output, e_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        dropout_mha: float = 0.0,
        dropout_ff: float = 0.0,
        activation_ff: str = "relu",
        layer_norm_eps: float = 1e-6,
        use_pos_att: bool = False,
    ) -> None:
        """
        The transformer decoder layer from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf). The layer
        is made up of two multi-head attention sublayers (self-attention and encoder-decoder cross-attention) followed
        by a feed-forward sublayer. Differently from the paper, this implementation uses pre-norm in the residual
        connection.
        :param d_model: the model's embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param dropout_mha: the dropout value for the multi-head attention (default=0.0).
        :param dropout_ff: the dropout value for the feed-forward sublayer (default=0.0).
        :param activation_ff: the activation function for the feed-forward sub-layer, can be either ReLU or GeLU
            (default="relu").
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        :param use_pos_att: whether to use a positional attention sublayer between the multi-head and cross
            attentions sublayers (default=False).
        """
        super().__init__()
        # Multi-head attention sublayer
        self.mha_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.mha_dropout = nn.Dropout(dropout)

        # Positional attention sublayer
        self._use_pos_att = use_pos_att
        if self._use_pos_att:
            self.pos_att_norm = nn.LayerNorm(d_model, layer_norm_eps)
            self.pos_att = MultiHeadAttention(d_model, n_heads, dropout_mha)
            self.pos_att_dropout = nn.Dropout(dropout)

        # Cross-attention sublayer
        self.ca_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ca = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.ca_dropout = nn.Dropout(dropout)

        # Feed-forward sublayer
        self.ff_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ff = FeedForwardLayer(d_model, dim_ff, dropout_ff, activation_ff)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_embeddings: torch.Tensor,
        e_output: torch.Tensor,
        d_mask: torch.Tensor = None,
        e_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Multi-head attention sublayer
        mha_out = self.mha_norm(tgt_embeddings)
        mha_out = self.mha(mha_out, mha_out, mha_out, d_mask)
        mha_out = tgt_embeddings + self.mha_dropout(mha_out)

        # Positional attention sublayer
        if self._use_pos_att:
            pos_att_out = self.pos_att_norm(mha_out)
            pos_att_out = self.pos_att(pos_att_out, pos_att_out, mha_out, d_mask)
            pos_att_out = self.pos_att_dropout(pos_att_out)
        else:
            pos_att_out = mha_out

        # Cross-attention sublayer
        ca_out = self.ca_norm(pos_att_out)
        ca_out = self.ca(ca_out, e_output, e_output, e_mask)
        ca_out = pos_att_out + self.mha_dropout(ca_out)

        # Feed-forward sublayer
        ff_out = self.ff_norm(ca_out)
        ff_out = self.ff(ff_out)
        out = ca_out + self.ff_dropout(ff_out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int = 6, norm: nn.LayerNorm = None):
        """
        The decoder from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf). Following the actual
        implementation of the paper, a LayerNorm layer is put at the end of the decoder layers stack.
        :param decoder_layer: transformer's decoder layer that will be used in order to build the stack of
            decoder layers.
        :param num_layers: the number of layers (default=6).
        :param norm: the layer normalization that should be at the end of the decoder layers stack (default=None).
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        tgt_embeddings: torch.Tensor,
        e_output: torch.Tensor,
        d_mask: torch.Tensor = None,
        e_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        output = tgt_embeddings
        for decoder_layer in self.layers:
            output = decoder_layer(output, e_output, d_mask, e_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
