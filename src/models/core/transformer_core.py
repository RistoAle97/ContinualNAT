import torch
from torch import nn
from torch.optim import AdamW
from lightning import LightningModule
from transformers import get_cosine_schedule_with_warmup
from src.models.core import CoreConfig
from src.modules import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer,\
    TransformerDecoder


class TransformerCore(LightningModule):

    def __init__(self, config: CoreConfig) -> None:
        """
        This class does not implement the forward method and should be used only as a base for the actual model's
        implementation.
        """
        super().__init__()
        # Parameters
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.num_encoder_layers = config.num_encoder_layers
        self.num_decoder_layers = config.num_decoder_layers
        self.dim_ff = config.dim_ff
        self.dropout = config.dropout
        self.dropout_mha = config.dropout_mha
        self.dropout_ff = config.dropout_ff
        self.activation_ff = config.activation_ff
        self.layer_norm_eps = config.layer_norm_eps

        # Token ids
        self.sos_token_id = config.sos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

        # Embeddings and positional encoder
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id)
        self.positional_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        self.embedding_scale = 1.0 if not config.scale_embeddings else self.d_model ** 0.5

        # Encoder
        encoder_norm = nn.LayerNorm(self.d_model, self.layer_norm_eps)
        encoder_layer = TransformerEncoderLayer(self.d_model, self.n_heads, self.dim_ff, self.dropout, self.dropout_mha,
                                                self.dropout_ff, self.activation_ff, self.layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers, norm=encoder_norm)

        # Decoder
        decoder_norm = nn.LayerNorm(self.d_model, self.layer_norm_eps)
        decoder_layer = TransformerDecoderLayer(self.d_model, self.n_heads, self.dim_ff, self.dropout, self.dropout_mha,
                                                self.dropout_ff, self.activation_ff, self.layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer, self.num_decoder_layers, norm=decoder_norm)

        # Linear output
        self.linear_output = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.linear_output.weight = self.embedding.weight

        # Train and validation losses for logging purposes
        self.batches_seen = 0
        self.train_metrics = {"train_loss": 0}
        self.val_metrics = {"val_loss": 0}

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

    def on_train_start(self) -> None:
        # Set logging metrics to initial value (this is necessary if training is resumed from a checkpoint)
        self.batches_seen = 0
        self.train_metrics = dict.fromkeys(self.train_metrics, 0)
        self.val_metrics = dict.fromkeys(self.val_metrics, 0)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self.batches_seen += 1
        batches = self.trainer.log_every_n_steps * self.trainer.accumulate_grad_batches
        if self.batches_seen % batches == 0:
            self.train_metrics = {key: value / batches for key, value in self.train_metrics.items()}
            self.log_dict(self.train_metrics, prog_bar=True)
            self.train_metrics = dict.fromkeys(self.train_metrics, 0)
        elif self.batches_seen == 1:
            self.log_dict(self.train_metrics, prog_bar=True)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        dataloader_size = len(self.trainer.val_dataloaders)
        if (batch_idx + 1) % dataloader_size == 0:
            self.val_metrics = {key: value / dataloader_size for key, value in self.val_metrics.items()}
            self.log_dict(self.val_metrics)
            self.val_metrics = dict.fromkeys(self.val_metrics, 0)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-4)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0, self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def generate(self, *kwargs):
        """
        Method for generating the translation's tokens given the tokenized source language sentence.
        """
        raise NotImplementedError