# ContinualNAT
Master Degree thesis on Continual learning for multilingual non-autoregressive neural machine translation (NAT).

## Built with
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)
[![Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white)](https://github.com/Lightning-AI/lightning)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-huggingface-FF9D00?style=for-the-badge&labelColor=FFD21E)](https://github.com/huggingface/transformers)

## Setting and datasets
- **Translation directions:** $en \Leftrightarrow de$, $en \Leftrightarrow fr$, $en \Leftrightarrow es$ (we opted for an english-centric setting). 
- **Tokenizer training set:** [CC100](https://huggingface.co/datasets/cc100).
- **Training set:** [CCMatrix](https://huggingface.co/datasets/yhavinga/ccmatrix).
- **Validation set:**
  - _newstest2012_ for $en \Leftrightarrow es$.
  - _newstest2013_ for $en \Leftrightarrow de$ and $en \Leftrightarrow fr$.
- **Test set:**
  - _newstest2013_ for $en \Leftrightarrow es$.
  - _newstest2014_ for $en \Leftrightarrow de$ and $en \Leftrightarrow fr$.

The validation and test sets are on a [personal public repository](https://huggingface.co/datasets/thesistranslation/wmt14) on the Huggingface hub.

## Models
The NAT models' names are taken from the following [survey](https://arxiv.org/pdf/2204.09269.pdf).
### AR (autoregressive) models
- [Transformer](https://arxiv.org/abs/1706.03762)

### NAR (non-autoregressive) models
- [FT-NAT](https://arxiv.org/abs/1711.02281) (_WIP_)
- [CMLM](https://arxiv.org/abs/1904.09324)
- [GLAT](https://arxiv.org/abs/2008.07905) (_WIP_)
