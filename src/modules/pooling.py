import torch
from torch import nn
from torch.functional import F


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


class Pooler(nn.Module):

    def __init__(self, d_model: int = 512):
        """
        Pooler layer similar to the one from BERT by Devlin et al. https://arxiv.org/pdf/1810.04805.pdf. While its main
        application is for classification tasks, it can also be used as length predictor as Ghazvininejad et al.
        https://arxiv.org/pdf/1904.09324.pdf.
        :param d_model: embedding dimension (default=512).
        """
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, e_output: torch.Tensor, length_prediction: bool = False) -> torch.Tensor:
        out = self.linear(e_output[:, 0])
        out = nn.functional.tanh(out)
        if length_prediction:
            out = F.log_softmax(out, -1)
            out = out.argmax(-1)

        return out
