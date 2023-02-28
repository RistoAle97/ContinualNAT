import torch
import math
from torch import nn
from torch.functional import F


def positional_encoding(x: torch.Tensor, d_model: int = 512, max_len: int = 5000) -> torch.Tensor:
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    x = x + pe[:, x.size(1)]  # (batch_size, seq_len, d_model)
    return x


class TransformerCore(nn.Module):

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
                 share_embeddings_src_tgt: bool = True,
                 share_embeddings_tgt_out: bool = True) -> None:
        """
        This class does not implement the forward method and should be used only as a base for the actual model's
        implementation.
        :param src_vocab_size: input language vocabulary size.
        :param tgt_vocab_size: target language vocabulary size, if no value is passed, then it will have the same size
            of the source one.
        :param d_model: embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param num_encoder_layers: the number of encoder layers (default=6).
        :param num_decoder_layers: the number of decoder layers (default=6).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-5).
        :param share_embeddings_src_tgt: whether to share the weights beetween source and target embedding layers
            (default=True).
        :param share_embeddings_tgt_out: whether to share the weights beetween the target embeddings and the linear
            output (default=True).
        """
        super().__init__()
        # Parameters
        self.src_vocab_size = src_vocab_size
        if tgt_vocab_size is not None and not share_embeddings_src_tgt:
            self.tgt_vocab_size = tgt_vocab_size
        else:
            self.tgt_vocab_size = src_vocab_size

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layer = num_decoder_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.share_embeddings_src_trg = share_embeddings_src_tgt
        self.share_embeddings_trg_out = share_embeddings_tgt_out

        # Embeddings
        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, d_model)
        if share_embeddings_src_tgt and tgt_vocab_size is not None:
            raise ValueError("Vocab size for the decoder was passed while the parameter for sharing the embeddings "
                             "between encoder and decoder was set to True.")

        if share_embeddings_src_tgt or tgt_vocab_size is None:
            self.tgt_embedding.weight = self.src_embedding.weight

        self.positional_dropout = nn.Dropout(dropout)

        # Encoder and decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, layer_norm_eps=layer_norm_eps,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_ff, dropout, layer_norm_eps=layer_norm_eps,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Linear output
        self.linear_output = nn.Linear(d_model, self.tgt_vocab_size, bias=False)
        if share_embeddings_tgt_out:
            self.linear_output.weight = self.tgt_embedding.weight


class Transformer(TransformerCore):

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
                 share_embeddings_src_tgt: bool = True,
                 share_embeddings_tgt_out: bool = True) -> None:
        """
        Transformer model whose architecture is based on the paper "Attention is all you need" from Vaswani et al.
        https://arxiv.org/pdf/1706.03762.pdf. The model, differently from the pytorch implementation, comes with
        embeddings, positional encoding, linear output and softmax layers. The model expects inputs with the format
        (batch_size, seq_len).
        :param src_vocab_size: input language vocabulary size.
        :param tgt_vocab_size: target language vocabulary size, if no value is passed, then it will have the same size
            of the source one.
        :param d_model: embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param num_encoder_layers: the number of encoder layers (default=6).
        :param num_decoder_layers: the number of decoder layers (default=6).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-5).
        :param share_embeddings_src_tgt: whether to share the weights beetween source and target embedding layers
            (default=True).
        :param share_embeddings_tgt_out: whether to share the weights beetween the target embeddings and the linear
            output (default=True).
        """
        super().__init__(src_vocab_size, tgt_vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers,
                         dim_ff, dropout, layer_norm_eps, share_embeddings_src_tgt, share_embeddings_tgt_out)

    def encode(self,
               src_input: torch.Tensor,
               e_mask: torch.Tensor = None,
               e_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encodes the masked source sentence.
        :param src_input: torch tensor of shape (batch_size, seq_len).
        :param e_mask: causal mask for the encoder of shape (seq_len, seq_len).
        :param e_pad_mask: key padding mask for the encoder of shape (batch_size, seq_len).
        :return: torch tensor representing the encodings with shape (batch_size, seq_len, d_model).
        """
        src_input = self.src_embedding(src_input)
        src_input = positional_encoding(src_input)
        src_input = self.positional_dropout(src_input)
        e_output = self.encoder(src_input, e_mask, e_pad_mask)
        return e_output

    def decode(self,
               e_output: torch.Tensor,
               tgt_input: torch.Tensor,
               e_mask: torch.Tensor = None,
               d_mask: torch.Tensor = None,
               e_pad_mask: torch.Tensor = None,
               d_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Decodes the masked target sentence given the encodings of the source sentence.
        :param e_output: encodings coming from the encoder of shape (batch_size, seq_len, d_model).
        :param tgt_input: torch tensor of shape (batch_size, seq_len)
        :param e_mask: causal mask for the encoder of shape (seq_len, seq_len).
        :param d_mask: causal mask for the decoder of shape (seq_len, seq_len).
        :param e_pad_mask: key padding mask for the encoder of shape (batch_size, seq_len) used for the multi-head
            encoder-decoder attention.
        :param d_pad_mask: key padding mask for the decoder of shape (batcg_size, seq_len).
        :return: torch tensor representing the decodings with shape (batch_size, seq_len, d_model).
        """
        tgt_input = self.tgt_embedding(tgt_input)
        tgt_input = positional_encoding(tgt_input)
        tgt_input = self.positional_dropout(tgt_input)
        d_output = self.decoder(e_output, tgt_input, e_output, d_mask, e_mask, d_pad_mask, e_pad_mask)
        return d_output

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
        # Embeddings and positional encoding
        src_input = self.src_embedding(src_input)  # (batch_size, seq_len, d_model)
        tgt_input = self.tgt_embedding(tgt_input)  # (batch_size, seq_len, d_model)
        src_input = positional_encoding(src_input, self.d_model)
        tgt_input = positional_encoding(tgt_input, self.d_model)
        src_input = self.positional_dropout(src_input)  # (batch_size, seq_len, d_model)
        tgt_input = self.positional_dropout(tgt_input)  # (batch_size, seq_len, d_model)

        # Encoder and decoder
        e_output = self.encoder(src_input, e_mask, e_pad_mask)
        d_output = self.decoder(tgt_input, e_output, d_mask, e_mask, d_pad_mask, e_pad_mask)

        # Linear output and softmax
        output = self.linear_output(d_output)  # (batch_size, seq_len, tgt_vocab_size)
        output = F.log_softmax(output, -1)
        return output
