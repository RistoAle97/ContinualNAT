import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from ..modules import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer,\
    TransformerDecoder


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
        self.embedding_scale = 1.0 if not scale_embeddings else d_model ** 0.5

        # Encoder
        encoder_norm = nn.LayerNorm(d_model, layer_norm_eps)
        encoder_layer = TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, norm=encoder_norm)

        # Decoder
        decoder_norm = nn.LayerNorm(d_model, layer_norm_eps)
        decoder_layer = TransformerDecoderLayer(d_model, n_heads, dim_ff, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, norm=decoder_norm)

        # Linear output
        self.linear_output = nn.Linear(d_model, self.vocab_size, bias=False)
        self.linear_output.weight = self.embedding.weight

        # Train and validation losses
        self.train_loss = 0
        self.val_loss = 0

    def encode(self,
               e_input: torch.Tensor,
               e_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encodes the masked source sentence.
        :param e_input: torch tensor of shape (bsz, seq_len).
        :param e_mask: mask for the encoder of shape (bsz, seq_len).
        :return: torch tensor representing the encodings with shape (bsz, seq_len, d_model).
        """
        src_embeddings = self.embedding(e_input)  # (bsz, seq_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)
        e_output = self.encoder(src_embeddings, e_mask)
        return e_output

    def decode(self,
               tgt_input: torch.Tensor,
               e_output: torch.Tensor,
               d_mask: torch.Tensor = None,
               e_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Decodes the masked target sentence given the encodings of the source sentence.
        :param e_output: encodings coming from the encoder of shape (bsz, seq_len, d_model).
        :param tgt_input: torch tensor of shape (bsz, seq_len)
        :param e_mask: mask for the encoder of shape (bsz, seq_len).
        :param d_mask: mask for the decoder of shape (bsz, seq_len).
        :return: torch tensor representing the decodings with shape (bsz, seq_len, vocab_size).
        """
        tgt_embeddings = self.embedding(tgt_input)  # (bsz, seq_len, d_model)
        tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, e_mask)
        d_output = self.linear_output(d_output)  # (bsz, seq_len, vocab_size)
        return d_output

    def compute_loss(self, *kwargs) -> torch.Tensor:
        """
        Method for computing the model's loss.
        """
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-4)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0, self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def generate(self, *kwargs):
        """
        Method for generating the translation's tokens given the tokenized source language sentence.
        """
        raise NotImplementedError
