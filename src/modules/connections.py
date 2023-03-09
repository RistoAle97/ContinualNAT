import torch
from torch import nn
from torch.functional import F


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float = 0.1) -> None:
        """
        Represents a residual connection around a layer.
        :param dropout: dropout value (default=0.1).
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the sum between the residual from before the layer and its output.
        :param residual: the residual coming from before the layer.
        :param x: the layer's output.
        :return: tensor representing the residual connection's output.
        """
        return residual + self.dropout(x)


class HighwayConnection(nn.Module):

    def __init__(self, d_model: int = 512, dropout: float = 0.1) -> None:
        """
        Represents a highway connection around a layer.
        :param d_model: the model's embedding dimension (default=512).
        :param dropout: dropout value (default=0.1).
        """
        super().__init__()
        self.highway = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the sum between the residual from before the layer and its output. For doing so, the connection
        computes a probability by linear transforming the residual and, then, performs a weighted sum between
        the residual and the layer's output.
        :param residual: the residual coming from before the layer.
        :param x: the layer's output.
        :return: tensor representing the highway connection's output, residual * (1 - p) + x * p, where p is
            the probability given by the linear transformation of the residual.
        """
        p = F.sigmoid(self.highway(residual))
        x = self.dropout(x)
        return residual * (1 - p) + x * p
