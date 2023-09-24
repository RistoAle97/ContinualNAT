import torch
from torch.functional import F

from continualnat.models.core.transformer_core import TransformerCore
from continualnat.models.transformer.config_transformer import TransformerConfig
from continualnat.inference.strategies import greedy_decoding, beam_decoding
from continualnat.utils.masks import create_masks
from continualnat.utils.models import init_bert_weights


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

    def forward(
        self,
        src_input: torch.Tensor,
        tgt_input: torch.Tensor,
        e_mask: torch.Tensor = None,
        d_mask: torch.Tensor = None
    ) -> torch.Tensor:
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
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_token_id, label_smoothing=self.label_smoothing)
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

        # Update metrics for logging
        self.train_metrics["train_loss"].update(loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        input_ids = batch["input_ids"]
        references = batch["references"]

        # Compute translations
        tokenizer, lang_pair, tgt_lang = super()._val_tokenizer_tgtlang(dataloader_idx)
        translation = self.generate(input_ids, tokenizer.lang_code_to_id[tgt_lang], num_beams=1)
        predictions = tokenizer.batch_decode(translation, skip_special_tokens=True)

        # Update the BLEU metric internal parameters
        self.val_metrics[f"BLEU_{lang_pair}"].update(predictions, references)

    def generate(
        self,
        input_ids: torch.Tensor,
        decoder_start_token_id: int,
        max_new_tokens: int = 10,
        num_beams: int = 5
    ) -> torch.Tensor:
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
