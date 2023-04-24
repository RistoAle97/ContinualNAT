import torch
from torch.functional import F
from src.models import TransformerCore, TransformerConfig
from src.inference import greedy_decoding, beam_decoding
from src.utils import init_bert_weights, create_masks


class Transformer(TransformerCore):

    def __init__(self, config: TransformerConfig) -> None:
        """
        Transformer model whose architecture is based on the paper "Attention is all you need" from Vaswani et al.
        https://arxiv.org/pdf/1706.03762.pdf. The model, differently from the pytorch implementation, comes with
        embeddings, positional encoding, linear output and softmax layers. The model expects inputs with the format
        (bsz, seq_len).
        """
        super().__init__(config)
        # Initialize weights
        self.apply(init_bert_weights)

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                e_mask: torch.Tensor = None,
                d_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Process masked source and target sequences.
        """
        # Embeddings and positional encoding
        src_embeddings = self.embedding(src_input)  # (bsz, seq_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)
        tgt_embeddings = self.embedding(tgt_input)  # (bsz, seq_len, d_model)
        tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)

        # Encoder and decoder
        e_output = self.encoder(src_embeddings, e_mask)
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, e_mask)

        # Linear output
        output = self.linear_output(d_output)  # (bsz, seq_len, vocab_size)
        return output

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_token_id)
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        decoder_input_ids = batch["decoder_input_ids"]

        # Create masks
        e_mask, d_mask = create_masks(input_ids, decoder_input_ids, self.pad_token_id, "causal")

        # Compute loss
        logits = self(input_ids, decoder_input_ids, e_mask=e_mask, d_mask=d_mask)
        loss = self.compute_loss(logits, labels)

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
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        decoder_input_ids = batch["decoder_input_ids"]

        # Create masks
        e_mask, d_mask = create_masks(input_ids, decoder_input_ids, self.pad_token_id, "causal")

        # Compute loss
        logits = self(input_ids, decoder_input_ids, e_mask, d_mask)
        loss = self.compute_loss(logits, labels)

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
                 num_beams: int = 5) -> torch.Tensor:
        """
        Generate tokens at inference time using greedy or beam search decoding.
        :param input_ids: tokenized source sentence.
        :param decoder_start_token_id: the token that will propend the output sequence, in a multilingual setting this
            should be the target language token, in a blingual setting the start of sequence token should be
            used instead.
        :param max_new_tokens: the number of new tokens allowed on top of the source sentence length (default=10).
        :param num_beams: size of the beam, if it is equal to 1 than greedy decoding will be applied, otherwise
            beam search will be performed (default=5).
        :return: tokenized translation of the source sentence.
        """
        if num_beams < 1:
            raise ValueError("The beam size must be at least 1.")

        if max_new_tokens < 0:
            raise ValueError("The number of max new tokens must be at least 0.")

        self.eval()
        if num_beams == 1:
            output = greedy_decoding(self, input_ids, decoder_start_token_id, max_new_tokens)
        else:
            output = beam_decoding(self, input_ids, decoder_start_token_id, max_new_tokens, num_beams)

        return output
