from typing import Tuple, Union

import torch
from lightning import LightningModule
from torch import nn
from torch.optim import Optimizer, AdamW
from torchmetrics import MeanMetric
from torchmetrics.text import SacreBLEUScore
from transformers import MBartTokenizer, MBartTokenizerFast, get_scheduler

from src.data.datasets import TranslationDataset, IterableTranslationDataset
from src.models.core.config_core import CoreConfig
from src.modules.positional_encoding import PositionalEncoding
from src.modules.transformer_layers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer,\
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
        self.bos_token_id = config.bos_token_id
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

        # Optimizer and learning rate scheduler
        self.optimizer = AdamW(self.parameters(), lr=5e-4)
        self.lr_scheduler = {"name": "cosine", "num_warmup_steps": 0}

        # Label smoothing value
        self.label_smoothing = config.label_smoothing

        # Train loss and validation mean BLEU for logging purposes
        self.train_metrics = {"train_loss": MeanMetric()}
        self.val_metrics = {"mean_BLEU": MeanMetric()}

    def encode(self,
               e_input: torch.Tensor,
               e_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encodes the masked source sentence.
        :param e_input: torch tensor of shape (bsz, seq_len).
        :param e_mask: mask for the encoder of shape (bsz, 1, seq_len).
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
        :param e_mask: mask for the encoder of shape (bsz, 1, seq_len).
        :param d_mask: mask for the decoder of shape (bsz, seq_len, seq_len).
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

    def _val_tokenizer_tgtlang(self, dataloader_idx) -> Tuple[Union[MBartTokenizer, MBartTokenizerFast], str, str]:
        # Use the tokenizer from the dataloader's dataset
        langs_dataloader = list(self.trainer.val_dataloaders.items())[dataloader_idx]
        lang_pair, dataloader = langs_dataloader
        if isinstance(dataloader.dataset, Union[TranslationDataset, IterableTranslationDataset]):
            tokenizer = dataloader.dataset.tokenizer
            tgt_lang_code = dataloader.dataset.tgt_lang_code
        else:
            raise ValueError("You should use a TranslationDataset or IterableTranslationDataset as datasets for the"
                             "dataloader.")

        # Compute translations
        return tokenizer, lang_pair, tgt_lang_code

    def on_train_start(self) -> None:
        self.train_metrics = {metric_name: MeanMetric() for metric_name in self.train_metrics.keys()}

    def on_validation_start(self) -> None:
        langs_dataloaders = list(self.trainer.val_dataloaders.items())
        for lang_pair, _ in langs_dataloaders:
            metric_name = f"BLEU_{lang_pair}"
            if metric_name not in self.val_metrics.keys():
                self.val_metrics[metric_name] = SacreBLEUScore()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        batches = self.trainer.log_every_n_steps * self.trainer.accumulate_grad_batches
        if batch_idx == 0 and self.trainer.current_epoch == 0:
            # First batch seen during training
            metrics_to_log = {metric_name: value.compute() for metric_name, value in self.train_metrics.items()}
            self.logger.log_metrics(metrics_to_log, 0)
        if (batch_idx + 1) * (self.trainer.current_epoch + 1) % batches == 0:
            # Log every n optimizer steps
            metrics_to_log = {metric_name: metric.compute() for metric_name, metric in self.train_metrics.items()}
            self.log_dict(metrics_to_log, prog_bar=True)
            for metric in self.train_metrics.values():
                metric.reset()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        langs_dataloader = list(self.trainer.val_dataloaders.items())[dataloader_idx]
        lang_pair, dataloader = langs_dataloader
        if (batch_idx + 1) % len(dataloader) == 0:
            metric_name = f"BLEU_{lang_pair}"

            # Compute the BLEU score for the translation direction and log it
            bleu_score = self.val_metrics[metric_name].compute() * 100
            self.val_metrics["mean_BLEU"].update(bleu_score)
            if dataloader_idx == len(self.trainer.val_dataloaders) - 1:
                self.log("mean_BLEU", self.val_metrics["mean_BLEU"].compute(), prog_bar=True, add_dataloader_idx=False,
                         batch_size=dataloader.batch_size)
                self.val_metrics["mean_BLEU"].reset()

            self.log(metric_name, bleu_score, prog_bar=True, add_dataloader_idx=False, batch_size=dataloader.batch_size)
            self.val_metrics[metric_name].reset()

    def change_optimizer(self, optimizer: Optimizer) -> None:
        """
        Change the current model's optimizer. Keep in mind that the model uses AdamW with lr = 5e-4 as default.
        :param optimizer: the new optimizer.
        """
        self.optimizer = optimizer

    def change_lr_scheduler(self, lr_scheduler: str = None, warmup_steps: int = None) -> None:
        """
        Change the learning rate scheduler name or warmup steps. Keep in mind that the model uses a cosine scheduler
        with no warmup steps as default.
        :param lr_scheduler: the learning rate scheduler's name, such scheduler should be available in the Transformers
            library (default=None).
        :param warmup_steps: the number of warmup steps (default=None).
        """
        if lr_scheduler is not None:
            self.lr_scheduler["name"] = lr_scheduler

        if warmup_steps is not None and warmup_steps >= 0:
            self.lr_scheduler["num_warmup_steps"] = warmup_steps

    def configure_optimizers(self):
        lr_scheduler = get_scheduler(**self.lr_scheduler, optimizer=self.optimizer,
                                     num_training_steps=self.trainer.estimated_stepping_batches)
        return [self.optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def generate(self, *kwargs):
        """
        Method for generating the translation's tokens given the tokenized source language sentence.
        """
        raise NotImplementedError
