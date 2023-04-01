import torch
from torch.functional import F
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_inverse_sqrt_schedule
from .transformer_core import TransformerCore
from ..inference import greedy_decoding, beam_decoding
from ..utils import generate_causal_mask, init_bert_weights


class Transformer(TransformerCore):

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6) -> None:
        """
        Transformer model whose architecture is based on the paper "Attention is all you need" from Vaswani et al.
        https://arxiv.org/pdf/1706.03762.pdf. The model, differently from the pytorch implementation, comes with
        embeddings, positional encoding, linear output and softmax layers. The model expects inputs with the format
        (batch_size, seq_len).
        :param vocab_size: shared vocabulary size.
        :param d_model: embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param num_encoder_layers: the number of encoder layers (default=6).
        :param num_decoder_layers: the number of decoder layers (default=6).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        """
        super().__init__(vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers,
                         dim_ff, dropout, layer_norm_eps)
        # Initialize weights
        self.apply(init_bert_weights)

        # Scheduler
        # self.lr_scheduler = None
        self.train_loss = 0
        self.val_loss = 0

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                d_mask: torch.Tensor = None,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Process masked source and target sequences.
        """
        # Embeddings and positional encoding
        src_embeddings = self.embedding(src_input)  # (batch_size, seq_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)
        tgt_embeddings = self.embedding(tgt_input)  # (batch_size, seq_len, d_model)
        tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)

        # Encoder and decoder
        e_output = self.encoder(src_embeddings, None, e_pad_mask)
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, None, d_pad_mask, e_pad_mask)

        # Linear output
        output = self.linear_output(d_output)  # (batch_size, seq_len, vocab_size)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 4000, self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # noinspection DuplicatedCode
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        decoder_input_ids = batch["decoder_input_ids"]

        # Create masks
        e_pad_mask = (input_ids == self.pad_token_id).to(self.device)
        d_pad_mask = (decoder_input_ids == self.pad_token_id).to(self.device)
        # noinspection DuplicatedCode
        d_mask = generate_causal_mask(decoder_input_ids.shape[-1]).to(self.device)

        # Compute loss
        logits = self(input_ids, decoder_input_ids, d_mask, e_pad_mask, d_pad_mask)
        loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1),
                               ignore_index=self.pad_token_id)

        # Log train loss
        self.train_loss += loss.item()
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.log("train_loss", self.train_loss / self.trainer.log_every_n_steps, prog_bar=True)
            self.train_loss = 0
        elif self.trainer.global_step == 0:
            self.log("train_loss", self.train_loss, prog_bar=True)
            self.train_loss = 0

        return loss

    def validation_step(self, batch, batch_idx):
        # noinspection DuplicatedCode
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        decoder_input_ids = batch["decoder_input_ids"]

        # Create masks
        e_pad_mask = (input_ids == self.pad_token_id).to(self.device)
        d_pad_mask = (decoder_input_ids == self.pad_token_id).to(self.device)
        # noinspection DuplicatedCode
        d_mask = generate_causal_mask(decoder_input_ids.shape[-1]).to(self.device)

        # Compute loss
        logits = self(input_ids, decoder_input_ids, d_mask, e_pad_mask, d_pad_mask)
        loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1),
                               ignore_index=self.pad_token_id)

        # Log validation loss
        self.val_loss += loss.item()
        if (batch_idx + 1) % len(self.trainer.val_dataloaders) == 0:
            self.log("val_loss", self.val_loss / len(self.trainer.val_dataloaders), prog_bar=True)
            self.val_loss = 0

        return loss

    def generate(self,
                 input_ids: torch.Tensor,
                 decoder_start_token_id: int,
                 max_new_tokens: int = 10,
                 beam_size: int = 5) -> torch.Tensor:
        """
        Generate tokens at inference time using greedy or beam search decoding.
        :param input_ids: tokenized source sentence.
        :param decoder_start_token_id: the token that will propend the output sequence, in a multilingual setting this
            should be the target language token, in a blingual setting the start of sequence token should be
            used instead.
        :param max_new_tokens: the number of new tokens allowed on top of the source sentence length (default=10).
        :param beam_size: size of the beam, if it is equal to 1 than greedy decoding will be applied, otherwise
            beam search will be performed (default=5).
        :return: tokenized translation of the source sentence.
        """
        if beam_size < 1:
            raise ValueError("The beam size must be at least 1.")

        if max_new_tokens < 0:
            raise ValueError("The number of max new tokens must be at least 0.")

        if beam_size == 1:
            output = greedy_decoding(self, input_ids, decoder_start_token_id, max_new_tokens)
        else:
            output = beam_decoding(self, input_ids, self.sos_token_id, self.eos_token_id, self.pad_token_id,
                                   max_new_tokens, beam_size)

        return output
