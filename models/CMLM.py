import torch
from torch import nn
from models.transformer import Transformer


class CMLM(Transformer):

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
                 share_embeddings_tgt_out: bool = True) -> None:
        super().__init__(src_vocab_size, tgt_vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers,
                         dim_ff, dropout, layer_norm_eps, share_embeddings_src_tgt, share_embeddings_tgt_out)
        self.apply(self._init_bert_weigths)  # use BERT weight initialization

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
                e_mask: torch.Tensor = None,
                d_mask: torch.Tensor = None,
                e_pad_mask: torch.Tensor = None,
                d_pad_mask: torch.Tensor = None) -> torch.Tensor:
        return super().forward(src_input, tgt_input, e_mask, None, e_pad_mask, d_pad_mask)
