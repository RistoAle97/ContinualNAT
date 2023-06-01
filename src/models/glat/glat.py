import torch
from torch import nn
from torch.functional import F
from src.models.core import TransformerNATCore
from src.models.glat import GLATConfig
from src.modules import Pooler, DecoderLayerNAT, DecoderNAT
from src.utils import init_bert_weights, create_masks
from typing import Tuple


class GLAT(TransformerNATCore):

    def __init__(self, config: GLATConfig) -> None:
        """
        The Glancing Transformer (GLAT) from Qian et al. https://arxiv.org/pdf/2008.07905.pdf, a
        non-autoregressive model that use GLM (Glancing Language Model) to learn word interdependecy. The model's
        architecture is the same as the FT-NAT from Gu et al. https://arxiv.org/pdf/1711.02281.pdf, but it uses a
        <length> token to predict the target sentences' lengths as Ghazvininejad et al.
        https://arxiv.org/pdf/1904.09324.pdf did for the CMLM model.
        """
        super().__init__(config)
        # Token ids
        self.length_token_id = config.length_token_id

        # Pooler layer after the encoder to predict the target sentence length
        self.pooler = Pooler(self.d_model)

        # Decoder
        decoder_norm = nn.LayerNorm(self.d_model, self.layer_norm_eps)
        decoder_layer = DecoderLayerNAT(self.d_model, self.n_heads, self.dim_ff, self.dropout, self.dropout_mha,
                                        self.dropout_ff, self.activation_ff, self.layer_norm_eps, False)
        self.decoder = DecoderNAT(decoder_layer, self.num_decoder_layers, norm=decoder_norm)

        # Use BERT weight initialization
        self.apply(init_bert_weights)

    def encode(self, e_input: torch.Tensor, e_mask: torch.Tensor = None) -> torch.Tensor:
        self.__check_length_token(e_input)
        e_output = super().encode(e_input, e_mask)
        return e_output

    def decode(self,
               tgt_input: torch.Tensor,
               e_output: torch.Tensor,
               d_mask: torch.Tensor = None,
               e_mask: torch.Tensor = None) -> torch.Tensor:
        d_output = super().decode(tgt_input, e_output[:, 1:], d_mask, e_mask[:, :, 1:])
        return d_output

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                e_mask: torch.Tensor = None,
                d_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process source and target sequences.
        """
        if not self._check_length_token(src_input):
            raise ValueError("The token <length> is not used by one or more tokenized sentence, the model needs such"
                             "token to predict the target lengths.")

        # Embeddings and positional encoding
        src_embeddings = self.embedding(src_input)  # (bsz, seq_len, d_model)
        tgt_embeddings = self.embedding(tgt_input)  # (bsz, seq_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)
        tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)

        # Encoder and decoder
        e_output = self.encoder(src_embeddings, e_mask)
        predicted_lengths = self.pooler(e_output)  # (bsz, seq_len)
        d_output = self.decoder(tgt_embeddings, e_output[:, 1:], d_mask, e_mask[:, :, 1:])

        # Linear output
        output = self.linear_output(d_output)  # (bsz, seq_len, vocab_size)
        return output, predicted_lengths

    def compute_loss(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor,
                     predicted_lengths: torch.Tensor,
                     target_lengths: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # Logits loss
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        logits_loss = F.cross_entropy(logits, labels, ignore_index=self.pad_token_id,
                                      label_smoothing=self.label_smoothing)

        # Length loss
        predicted_lengths = predicted_lengths.contiguous().view(-1, predicted_lengths.size(-1))
        target_lengths = target_lengths.contiguous().view(-1)
        lengths_loss = F.cross_entropy(predicted_lengths, target_lengths)

        # Combine the losses
        loss = logits_loss + lengths_loss
        return loss, logits_loss.item(), lengths_loss.item()

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        decoder_input_ids = batch["decoder_input_ids"]
        target_lengths = batch["target_lengths"]

        # Create masks
        e_mask, d_mask = create_masks(input_ids, decoder_input_ids, self.pad_token_id, None)

        # Compute loss
        logits, predicted_lengths = self(input_ids, decoder_input_ids, e_mask=e_mask, d_mask=d_mask)
        loss, logits_loss, lengths_loss = self.compute_loss(logits, labels, predicted_lengths, target_lengths)

        # Update train metrics
        self.train_metrics["train_loss"].update(loss.item())
        self.train_metrics["cmlm_mlm_loss"].update(logits_loss)
        self.train_metrics["cmlm_lengths_loss"].update(lengths_loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        input_ids = batch["input_ids"]
        references = batch["references"]

        # Compute translations
        tokenizer, lang_pair, tgt_lang = self._val_tokenizer_tgtlang(dataloader_idx)
        translation = self.generate(input_ids, tokenizer.lang_code_to_id[tgt_lang])
        predictions = tokenizer.batch_decode(translation, skip_special_tokens=True)

        # Update the BLEU metric internal parameters
        self.val_metrics[f"BLEU_{lang_pair}"].update(predictions, references)

    def generate(self,
                 input_ids: torch.Tensor,
                 tgt_lang_token_id: int = None) -> torch.Tensor:
        """
        Generate tokens during inference by using the mask-predict algorithm by Ghazvininejad et al.
        https://arxiv.org/pdf/1904.09324.pdf.
        :param input_ids: the tokenized source sentence.
        :param tgt_lang_token_id: the target language token id, if none is passed, then no token will be appended at the
            end of the target tokens (default=None).
        :return: tokenized translation of source sentence.
        """
        pass
