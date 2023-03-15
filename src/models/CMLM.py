import torch
import math
from torch import nn
from . import TransformerCore
from ..modules import Pooler
from typing import Tuple


class CMLM(TransformerCore):

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int = None,
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_ff: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6,
                 norm_first: bool = False,
                 share_embeddings_src_tgt: bool = True,
                 share_embeddings_tgt_out: bool = True) -> None:
        super().__init__(src_vocab_size, tgt_vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers,
                         dim_ff, dropout, layer_norm_eps, norm_first, share_embeddings_src_tgt,
                         share_embeddings_tgt_out)

        # Pooler layer after the encoder to predict the target sentence length
        self.pooler = Pooler(d_model)

        # Use BERT weight initialization
        self.apply(self._init_bert_weigths)

    @staticmethod
    def _init_bert_weigths(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,
                src_input: torch.Tensor,
                tgt_input: torch.Tensor,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process source and target sequences.
        """
        # Embeddings and positional encoding
        e_input = self.src_embedding(src_input)  # (batch_size, seq_len, d_model)
        d_input = self.tgt_embedding(tgt_input)  # (batch_size, seq_len, d_model)
        e_input = self.positional_encoder(e_input * math.sqrt(self.d_model))
        d_input = self.positional_encoder(d_input * math.sqrt(self.d_model))

        # Encoder and decoder
        e_output = self.encoder(e_input, None, e_pad_mask)
        predicted_length = self.pooler(e_output)  # (batch_size, seq_len)
        d_output = self.decoder(d_input, e_output, None, None, d_pad_mask, e_pad_mask)

        # Linear output
        output = self.linear_output(d_output)  # (batch_size, seq_len, tgt_vocab_size)
        return output, predicted_length

    def predict_target_length(self, e_output: torch.Tensor) -> torch.Tensor:
        predicted_length = self.pooler(e_output, True).unsqueeze(1)
        return predicted_length

    def _mask_predict(self,
                      input_ids: torch.Tensor,
                      sos_token_id: int,
                      pad_token_id: int,
                      mask_token_id: int,
                      max_iteration: int = 10) -> torch.Tensor:
        with torch.no_grad():
            # Parameters
            batch_size, seq_len = input_ids.size()
            device = next(self.parameters()).device

            # Encode the input tokens
            e_pad_mask = (input_ids == pad_token_id).to(device)
            encodings = self.encode(input_ids, e_pad_mask=e_pad_mask)

            # Predict the target lengths and take the largest one
            target_length = self.predict_target_length(encodings)
            max_predicted_length = target_length.max()

            output = torch.ones(batch_size, max_predicted_length.item(), dtype=torch.int,
                                device=device).fill_(pad_token_id)
            output[:, 0] = sos_token_id

    def generate(self,
                 input_ids: torch.Tensor,
                 sos_token_id: int,
                 pad_token_id: int,
                 mask_token_id: int,
                 iterations: int = 10) -> torch.Tensor:
        return self._mask_predict(input_ids, sos_token_id, pad_token_id, mask_token_id, iterations)
