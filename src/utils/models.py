from torch import nn
from typing import Tuple


def init_bert_weights(module: nn.Module) -> None:
    """
    Initialize module's weights following BERT https://arxiv.org/pdf/1810.04805.pdf.
    :param module: the module to initialize.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def model_n_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Computes the number of parameters, and the trainable ones, of a pytorch model.
    :param: model: a pytorch nn.Module.
    """
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_parameters = sum(p.numel() for p in model.parameters())
    return model_parameters, trainable_parameters


def model_size(model: nn.Module, size: str = "mb") -> float:
    """
    Computes the size of a pytorch model in terms of kb, mb or gb.
    :param model: a pytorch nn.Module.
    :param size: string which defines the wanted size to compute.
    :return: the size of the model.
    """
    if size not in ["kb", "mb", "gb"]:
        raise ValueError("The size of the model can only be shown in kb, mb or gb.")

    allowed_sizes = {"kb": 1, "mb": 2, "gb": 3}
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all = (param_size + buffer_size) / 1024 ** allowed_sizes[size]
    return size_all
