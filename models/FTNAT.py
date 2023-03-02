import torch
from torch import nn
from torch.functional import F
from models.TransformerCore import TransformerCore, positional_encoding


class Fertility(nn.Module):

    def __init__(self, d_model: int = 512, max_fertilities: int = 50):
        super().__init__()
        self.linear_output = nn.Linear(d_model, max_fertilities)

    def forward(self, e_output: torch.Tensor) -> torch.Tensor:
        fertilities = self.linear_output(e_output)  # (batch_size, seq_len, max_fertilities)
        fertilities = F.relu(fertilities)
        fertilities = F.log_softmax(fertilities, dim=-1)
        fertilities = torch.argmax(fertilities, dim=-1)  # (batch_size, seq_len)
        return fertilities


class FTNAT(TransformerCore):

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
                 share_embeddings_tgt_out: bool = True,
                 max_fertilities: int = 50) -> None:
        super().__init__(src_vocab_size, tgt_vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers,
                         dim_ff, dropout, layer_norm_eps, share_embeddings_src_tgt, share_embeddings_tgt_out)
        # Parameters
        self.max_fertilities = max_fertilities

        # Fertility
        self.fertility = Fertility(d_model, max_fertilities)

    @staticmethod
    def copy_fertilities(src_input: torch.Tensor, fertilities: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = src_input.shape
        copied_embedding = torch.zeros([batch_size, seq_len, d_model])
        for i, fertility_batch in enumerate(fertilities):
            pos = 0
            for j, fertility in enumerate(fertility_batch):
                if fertility == 0:
                    continue

                copied_embedding[i, pos:pos+int(fertility), :] = src_input[i, j, :].repeat(1, int(fertility), 1)
                pos += int(fertility)

        return copied_embedding

    def forward(self,
                src_input: torch.Tensor,
                e_mask: torch.Tensor = None,
                d_mask: torch.Tensor = None,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Embeddings and positional encoding
        e_embeddings = self.src_embedding(src_input)
        e_input = positional_encoding(e_embeddings, self.d_model)
        e_input = self.positional_dropout(e_input)

        # Encoder and fertilities
        e_output = self.encoder(e_input, e_mask, padding_mask)
        fertilities = self.fertility(e_output)
        copied_embeddings = self.copy_fertilities(e_embeddings, fertilities)

        # Decoder
        d_input = positional_encoding(copied_embeddings, self.d_model)
        d_input = self.positional_dropout(d_input)
        d_output = self.decoder.forward(d_input, e_output, d_mask, e_mask, padding_mask)

        # Linear output and softmax
        output = self.linear_output(d_output)  # (batch_size, seq_len, tgt_vocab_size)
        output = F.log_softmax(output, -1)
        return output

    def generate(self):
        pass
