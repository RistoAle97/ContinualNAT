import torch
from torch.functional import F
from torchmetrics import MeanMetric

from continualnat.models.core.config_nat_core import NATCoreConfig
from continualnat.models.core.transformer_core import TransformerCore
from continualnat.modules.pooling import LengthPooler, MeanPooler


class TransformerNATCore(TransformerCore):
    def __init__(self, config: NATCoreConfig) -> None:
        """
        This class does not implement the forward method and should be used only as a base for the actual NAT model's
        implementation.
        """
        super().__init__(config)
        # Parameters
        self.length_token_id = config.length_token_id
        self.map_copy = config.map_copy
        self.tensor_to_copy = config.tensor_to_copy
        self.pooler_size = config.pooler_size
        self.tau = config.tau

        # Pooler layer after the encoder to predict the target sentences' lengths
        if self.length_token_id is not None:
            self.pooler = LengthPooler(self.d_model, self.pooler_size)
        else:
            self.pooler = MeanPooler(self.d_model, self.pooler_size)

        # Length loss
        self.train_metrics["lengths_loss"] = MeanMetric()

    def encode(self, e_input: torch.Tensor, e_mask: torch.Tensor = None) -> torch.Tensor:
        if not self._check_length_token(e_input):
            raise ValueError(
                "The token <length> is not used by one or more tokenized sentence, the model needs such token to"
                "predict the target lengths."
            )

        e_output = super().encode(e_input, e_mask)
        return e_output

    def decode(
        self,
        tgt_input: torch.Tensor,
        e_output: torch.Tensor,
        d_mask: torch.Tensor = None,
        e_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.length_token_id is not None:
            # Do not use the encodings of the <length> token inside the decoder
            e_output = e_output[:, 1:]
            e_mask = e_mask[:, :, 1:]

        d_output = super().decode(tgt_input, e_output, d_mask, e_mask)
        return d_output

    def _define_pooler_inputs(
        self, e_output: torch.Tensor, e_mask: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        pooler_inputs = {"e_output": e_output}
        if self.length_token_id is not None:
            # Do not use the encodings of the <length> token inside the decoder
            e_output = e_output[:, 1:]
            e_mask = e_mask[:, :, 1:]
        else:
            # Use the encoder mask in the MeanPooler if no <length> token is defined
            pooler_inputs["e_mask"] = e_mask

        return pooler_inputs, e_output, e_mask

    def __uniform_copy(
        self,
        tensor_to_copy: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        The uniform copy mechanism from Gu et al. https://arxiv.org/pdf/1711.02281.pdf, it copies the source embeddings
        or encoder output as the decoder input based on the source and target lengths.
        :param tensor_to_copy: the source embeddings or the encoder output.
        :param src_lengths: the source sentences lengths taking into consideration the special tokens of shape (bsz, 1).
        :param tgt_lengths: the target sentences lengths taking into consideration the special tokens of shape (bsz, 1).
        :return: the decoder input ids.
        """
        max_tgt_len = tgt_lengths.max()
        bsz = tensor_to_copy.size(0)
        mask = torch.ones(bsz, max_tgt_len)
        for i, current_length in enumerate(tgt_lengths):
            mask[i, current_length:] -= 1

        mask = mask.bool()
        steps = (src_lengths.float() - 1) / (tgt_lengths.float() - 1 + 1e-4)  # step-size of shape (bsz)
        index_t = torch.arange(max_tgt_len, device=tgt_lengths.device).float()  # indexes of shape (max_tgt_len)
        index_t = steps[:, None] * index_t[None, :]  # (bsz, max_tgt_len)
        index_t = torch.round(index_t.squeeze(1)).long().detach()
        mapped_inputs = index_t.masked_fill(~mask, 0).to(tgt_lengths.device)
        mapped_inputs = mapped_inputs.unsqueeze(-1)
        embeddings_copy = torch.gather(
            tensor_to_copy, 1, mapped_inputs.expand(*mapped_inputs.size()[:-1], self.d_model)
        )
        return embeddings_copy

    def __soft_copy(
        self,
        tensor_to_copy: torch.Tensor,
        src_lengths: torch.tensor,
        tgt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        The soft copy mechanism from Wei et al. https://aclanthology.org/P19-1125.pdf, it copies the source embeddings
        or encoder output on an attention-based mechanism. Based on
        https://github.com/baoy-nlp/Latent-GLAT/blob/main/nat/vanilla_nat.py#L26.
        :param tensor_to_copy: the source embeddings or the encoder output.
        :param src_lengths: the source sentences lengths taking into consideration the special tokens of shape (bsz, 1).
        :param tgt_lengths: the target sentences lengths taking into consideration the special tokens of shape (bsz, 1).
        :return: the decoder input ids.
        """
        max_src_length = src_lengths.max().unsqueeze(-1)
        max_tgt_length = tgt_lengths.max().unsqueeze(-1)
        index_s = (
            torch.arange(max_src_length[-1], device=src_lengths.device).expand(*max_src_length).contiguous().float()
        )  # (max_src_length)
        index_t = (
            torch.arange(max_tgt_length[-1], device=tgt_lengths.device).expand(*max_tgt_length).contiguous().float()
        )  # (max_tgt_length)
        diff = -(index_t[:, None] - index_s[None, :]).abs()  # (max_tgt_length, max_src_length)
        diff = diff.unsqueeze(0).expand(tgt_lengths.size(0), *diff.size())  # (bsz, max_tgt_length, max_src_length)
        mask = (src_lengths[:, None] - 1 - index_s[None, :]).lt(0).float().squeeze(1)  # (bsz, max_src_length)
        logits = diff / self.tau - 1e9 * mask[:, None, :]
        probs = logits.softmax(-1)  # (bsz, max_tgt_length, max_src_length)
        embeddings_copy = torch.bmm(probs, tensor_to_copy)  # (bsz, max_tgt_length, d_model)
        return embeddings_copy

    def _copy_embeddings(
        self, tensor_to_copy: torch.Tensor, src_lengths: torch.Tensor, tgt_lengths: torch.Tensor
    ) -> torch.Tensor:
        if self.map_copy == "uniform":
            copied_tensor = self.__uniform_copy(tensor_to_copy, src_lengths, tgt_lengths)
        else:
            copied_tensor = self.__soft_copy(tensor_to_copy, src_lengths, tgt_lengths)

        return copied_tensor

    def _check_length_token(self, input_ids: torch.Tensor) -> bool:
        if self.length_token_id is None:
            # The model is not using a length token, we do not need to do any check
            return True

        is_using_length_token = input_ids[:, 0] == self.length_token_id
        return is_using_length_token.all()

    def predict_target_length(
        self,
        e_output: torch.Tensor,
        n_lengths: int = 1,
        e_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the target sentence possible lengths given the encoder's output.
        :param e_output: the encoder's output of shape (bsz, seq_len, d_model).
        :param n_lengths: the number of possible lengths to consider for each sentence (default=1).
        :param e_mask: mask for the encoder, you can leave this parameter as None if you are using the <length> token
            or if you want to perform mean pooling on all the source tokens (default=None).
        :return: the encodings of the target sentences' lengths.
        """
        pooler_inputs = {"e_output": e_output}
        if self.length_token_id is None:
            pooler_inputs["e_mask"] = e_mask

        lengths_logits = self.pooler(**pooler_inputs)  # (bsz, pooler_size)
        lengths_logits = F.log_softmax(lengths_logits, dim=-1)  # (bsz, pooler_size)
        lengths = lengths_logits.topk(n_lengths, dim=-1)[1]  # (bsz, n_lengths)
        return lengths

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lengths_logits: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        # Logits loss
        logits = logits.contiguous().view(-1, logits.size(-1))  # (bsz * seq_len, d_model)
        labels = labels.contiguous().view(-1)  # (bsz * seq_len)
        logits_loss = F.cross_entropy(
            logits, labels, ignore_index=self.pad_token_id, label_smoothing=self.label_smoothing
        )

        # Length loss
        lengths_logits = lengths_logits.contiguous().view(-1, lengths_logits.size(-1))
        target_lengths = target_lengths.contiguous().view(-1)
        lengths_loss = F.cross_entropy(lengths_logits, target_lengths)

        # Combine the losses
        loss = logits_loss + lengths_loss
        return loss, logits_loss.item(), lengths_loss.item()
