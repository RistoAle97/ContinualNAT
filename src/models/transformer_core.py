import torch
import math
import pytorch_lightning as pl
from torch import nn
from torch.functional import F
from ..modules import PositionalEncoding


class TransformerCore(pl.LightningModule):

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6,
                 scale_embeddings: bool = False,
                 sos_token_id: int = 0,
                 eos_token_id: int = 2,
                 pad_token_id: int = 1) -> None:
        """
        This class does not implement the forward method and should be used only as a base for the actual model's
        implementation.
        :param vocab_size: shared vocabulary size.
        :param d_model: embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param num_encoder_layers: the number of encoder layers (default=6).
        :param num_decoder_layers: the number of decoder layers (default=6).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        :param scale_embeddings: whether to scale the output of the embedding layer (default=False).
        :param sos_token_id: the start of sequence token id (default=0).
        :param eos_token_id: the end of sequence token id (default=2).
        :param pad_token_id: the pad token id (default=1).
        """
        super().__init__()
        # Parameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layer = num_decoder_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

        # Token ids
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Embeddings and positional encoder
        self.embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.embedding_scale = 1.0 if not scale_embeddings else math.sqrt(d_model)

        # Encoder and decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, activation=F.gelu,
                                                   layer_norm_eps=layer_norm_eps, batch_first=True, norm_first=True)
        encoder_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm=encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_ff, dropout, activation=F.gelu,
                                                   layer_norm_eps=layer_norm_eps, batch_first=True, norm_first=True)
        decoder_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, norm=decoder_norm)

        # Linear output
        self.linear_output = nn.Linear(d_model, self.vocab_size, bias=False)
        self.linear_output.weight = self.embedding.weight

    def encode(self,
               e_input: torch.Tensor,
               e_mask: torch.Tensor = None,
               e_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encodes the masked source sentence.
        :param e_input: torch tensor of shape (batch_size, seq_len).
        :param e_mask: causal mask for the encoder of shape (seq_len, seq_len).
        :param e_pad_mask: key padding mask for the encoder of shape (batch_size, seq_len).
        :return: torch tensor representing the encodings with shape (batch_size, seq_len, d_model).
        """
        src_embeddings = self.embedding(e_input)  # (batch_size, seq_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)
        e_output = self.encoder(src_embeddings, e_mask, e_pad_mask)
        return e_output

    def decode(self,
               e_output: torch.Tensor,
               tgt_input: torch.Tensor,
               d_mask: torch.Tensor = None,
               e_pad_mask: torch.Tensor = None,
               d_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Decodes the masked target sentence given the encodings of the source sentence.
        :param e_output: encodings coming from the encoder of shape (batch_size, seq_len, d_model).
        :param tgt_input: torch tensor of shape (batch_size, seq_len)
        :param d_mask: causal mask for the decoder of shape (seq_len, seq_len).
        :param e_pad_mask: key padding mask for the encoder of shape (batch_size, seq_len) used for the multi-head
            encoder-decoder attention.
        :param d_pad_mask: key padding mask for the decoder of shape (batcg_size, seq_len).
        :return: torch tensor representing the decodings with shape (batch_size, seq_len, vocab_size).
        """
        tgt_embeddings = self.embedding(tgt_input)  # (batch_size, seq_len, d_model)
        tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, None, d_pad_mask, e_pad_mask)
        d_output = self.linear_output(d_output)  # (batch_size, seq_len, vocab_size)
        return d_output

    def generate(self, *kwargs):
        """
        Method for generating the translation's tokens given the tokenized source language sentence.
        """
        raise NotImplementedError
