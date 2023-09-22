import torch
from torch import nn


class Pooler(nn.Module):

    def __init__(self, d_model: int = 512, pooler_size: int = 256) -> None:
        """
        Simple base class for all the pooler layers.
        :param d_model: the model's embedding dimension (default=512).
        :param pooler_size: the pooler hidden dimension, be sure that this is at least equal or higher than the
            tokenizer's max_length (default=256).
        """
        super().__init__()
        self.d_model = d_model
        self.pooler_size = pooler_size
        self.linear = nn.Linear(self.d_model, self.pooler_size, bias=False)


class LengthPooler(Pooler):

    def __init__(self, d_model: int = 512, pooler_size: int = 256) -> None:
        """
        Pooler layer similar to the one from BERT by Devlin et al. https://arxiv.org/pdf/1810.04805.pdf. While its main
        application is for classification tasks, it can also be used as a length predictor as Ghazvininejad et al.
        https://arxiv.org/pdf/1904.09324.pdf.
        :param d_model: embedding dimension (default=512).
        :param pooler_size: the pooler hidden dimension, be sure that this is at least equal or higher than the
            tokenizer's max_length (default=256).
        """
        super().__init__(d_model, pooler_size)

    def forward(self, e_output: torch.Tensor) -> torch.Tensor:
        out = self.linear(e_output[:, 0])  # (bsz, pooler_size)
        return out


class MeanPooler(Pooler):

    def __init__(self, d_model: int = 512, pooler_size: int = 256) -> None:
        """
        Mean pooler that works on the entire encoder's output.
        :param d_model: the model's embedding dimension (deafult=512).
        :param pooler_size: the pooler hidden dimension, be sure that this is at least equal or higher than the
            tokenizer's max_length (default=256).
        """
        super().__init__(d_model, pooler_size)

    def forward(self, e_output: torch.Tensor, e_mask: torch.Tensor = None) -> torch.Tensor:
        if e_mask is None:
            out = e_output.mean(1)  # (bsz, d_model)
        else:
            src_mask = e_mask.squeeze(1)  # (bsz, seq_len)
            out = (e_output / src_mask.sum(1)[:, None, None])  # (bsz, seq_len, d_model)
            out = (out * src_mask[:, :, None]).sum(1)  # (bsz, d_model)

        out = self.linear(out)  # (bsz, pooler_size)
        return out
