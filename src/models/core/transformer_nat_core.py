import torch
from torch.functional import F
from torchmetrics import MeanMetric
from src.models.core import TransformerCore, NATCoreConfig


class TransformerNATCore(TransformerCore):

    def __init__(self, config: NATCoreConfig) -> None:
        """
        This class does not implement the forward method and should be used only as a base for the actual NAT model's
        implementation.
        """
        super().__init__(config)
        # Parameters
        self.length_token_id = config.length_token_id
        self.src_embedding_copy = config.src_embedding_copy

        # Length loss
        self.train_metrics["lengths_loss"] = MeanMetric()

    def __uniform_copy(self, src_embeddings: torch.Tensor, e_mask: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
        """
        The uniform copy mechanism from Gu et al. https://arxiv.org/pdf/1711.02281.pdf, it copies the encoder's output
        as the decoder input based on the source and target lengths.
        :param src_embeddings: the source embeddings.
        :param e_mask: the encoder mask.
        :param d_mask: the decoder mask.
        :return: the decoder input ids.
        """
        src_lengths = e_mask.sum(dim=-1).view(-1)
        tgt_lengths = d_mask.sum(dim=-1)[:, 0]
        max_tgt_len = tgt_lengths.max()
        steps = (src_lengths.float() - 1) / (tgt_lengths.float() - 1)  # step-size of shape (batch_size)
        index_t = torch.arange(max_tgt_len, device=tgt_lengths.device).float()  # indexes of shape (max_tgt_len)
        index_t = steps[:, None] * index_t[None, :]  # (batch_size, max_tgt_len)
        index_t = torch.round(index_t).long().detach()
        mapped_inputs = index_t.masked_fill(~d_mask[:, 0, :], 0).to(tgt_lengths.device)
        mapped_inputs = mapped_inputs.unsqueeze(-1)
        embeddings_copy = torch.gather(src_embeddings, 1, mapped_inputs.expand(*mapped_inputs.size(), self.d_model))
        return embeddings_copy

    def __soft_copy(self, e_output: torch.Tensor, e_mask: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
        pass

    def _copy_embeddings(self,
                         src_embeddings: torch.Tensor,
                         e_mask: torch.Tensor,
                         d_mask: torch.Tensor) -> torch.Tensor:
        if self.src_embedding_copy == "uniform":
            return self.__uniform_copy(src_embeddings, e_mask, d_mask)
        else:
            return self.__soft_copy(src_embeddings, e_mask, d_mask)

    def _check_length_token(self, input_ids: torch.Tensor) -> bool:
        is_using_length_token = (input_ids[:, 0] == self.length_token_id)
        return is_using_length_token.all()

    def predict_target_length(self, e_output: torch.Tensor, n_lengths: int = 1) -> torch.Tensor:
        """
        Computes the target sentence possible lengths given the encoder's output.
        :param e_output: the encoder's output.
        :param n_lengths: the number of possible lengths to consider for each sentence.
        :return: the encodings of the target sentence length.
        """
        length_logits = self.pooler(e_output)
        length_logits = F.log_softmax(length_logits, dim=-1)
        lengths = length_logits.topk(n_lengths, dim=-1)[1]
        return lengths
