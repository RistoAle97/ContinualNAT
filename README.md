# ContinualNAT
Continual learning for multilingual non-autoregressive neural machine translation (NAT).

## Built with
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)
[![Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white)](https://github.com/Lightning-AI/lightning)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-huggingface-FF9D00?style=for-the-badge&labelColor=FFD21E)](https://github.com/huggingface/transformers)

## Setting and datasets
- **Translation directions:** en-de, de-en, en-fr, fr-en, en-es, es-en (we opted for an english-centric setting). 
- **Tokenizer training set:** [CC100](https://huggingface.co/datasets/cc100).
- **Training set:** [CCMatrix](https://huggingface.co/datasets/yhavinga/ccmatrix).
- **Validation set:** [Flores-200](https://huggingface.co/datasets/facebook/flores).
- **Test set:** For the en-es and es-en pairs the [newtest2013]() was used, while the [WMT14](https://huggingface.co/datasets/wmt14) dataset was chosen for all the other translation directions.

## Models
The NAT models' names are taken from the following [survey](https://arxiv.org/pdf/2204.09269.pdf).
### AR (autoregressive) models
- [Transformer](https://arxiv.org/abs/1706.03762)

### NAR (non-autoregressive) models
- [FT-NAT](https://arxiv.org/abs/1711.02281) (_WIP_)
- [RefineNAT](https://arxiv.org/abs/1802.06901) (_WIP_)
- [CMLM](https://arxiv.org/abs/1904.09324)
