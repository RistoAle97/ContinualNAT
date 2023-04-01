from torch import nn
from typing import Tuple
# from ..models import TransformerCore


def init_bert_weights(module: nn.Module) -> None:
    """
    Initialize module's weights following BERT https://arxiv.org/pdf/1810.04805.pdf. This method should be called by
    using self.apply(init_bert_weigths) inside the module class. The weigths of the nn.Linear and nn.Embedding layers
    are sampled by a normal distribution with mean 0.0 and std 0.02, while the weigths of nn.LayerNorm layer are set
    to 1.0. The bias of the nn.Linear, nn.LayerNorm layers and the weigths related to the padding token inside the
    nn.Embedding are then set to 0.
    :param module: the pytorch nn.Module to initialize.
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


'''def init_transformer_weights(module: nn.Module) -> None:
    """
    Initialize module's weigths following the transformer implementation from tensor2tensor. This method should be
    called by using self.apply(init_transformer_weigths) inside the module class. All the weights are initialized by
    using the Glorot initialization, except for the nn.Embedding of a TransformerCore module, in which case the weigths
    are initiliazed following a normal distribution with mean 0.0 and std equal to the inverse square root of the
    embedding dimension. If you are going to use this initialization it's greatly suggested to set the scale_embeddings
    parameter of a TransformerCore module to True.
    :param module: the pytorch nn.Module to initialize.
    """
    if isinstance(module, nn.Embedding):
        # module.weight.data.normal_(mean=0.0, std=module.d_model ** (- 0.5))
        module.weight.data.fill_(1.0)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    else:
        for p in module.parameters(False):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)'''


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
