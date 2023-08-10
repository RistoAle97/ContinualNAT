<div align="center">

# ContinualNAT
**M.Sc. thesis on Continual Learning for multilingual non-autoregressive Neural Machine Translation (NAT).**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)]()

[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)
[![Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white)](https://github.com/Lightning-AI/lightning)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-huggingface-FF9D00?style=for-the-badge&labelColor=FFD21E)](https://github.com/huggingface/transformers)

---
</div>

## :pushpin: Abstract
The Transformer architecture changed the world of Natural Language Processing and Deep Learning in general by setting new state-of-the-art scores for many fields and, nowadays, it is the go-to solution when approaching a new problem, but it comes with a limitation: its inference speed. The Transformer uses the parallelizable mechanism of self-attention during training in order to avoid the typical recurrence of RNN, but the use of an autoregressive (AR) decoder limits its full potential at inference time: at each time-step, only one token is generated.

In order to reach the full potential of the Transformer architecture a new kind of non-autoregressive (NAR) models were introduced , but it turned out that their performances were (and still are) way behind their AR counterparts.

Our purpose is to investigate the goodness of the one of the most famous NAR models in multilingual Neural Machine Translation (NMT) setting, while also testing their behaviour under a simple Continual Learning approach.

---
## :beginner: How to start
First, clone this repository
```bash
git clone https://github.com/RistoAle97/ContinualNAT
cd ContinualNAT
```

It is highly advised to create a new python virtual environment
```bash
pip install venv
python -m venv ContinualNAT
source ContinualNAT/bin/activate
```
or a conda environment before proceeding
```bash
conda create --name ContinualNAT
conda activate ContinualNAT
```

Then, install all the requirements
```bash
pip install -r requirements.txt
```

You can take a look at `train.py` to get an understanding of how to work with this repository, modify what you need and use
```bash
python train.py
```
to train one of the available models.

---

## :card_file_box: Setting and datasets
- **Translation directions:** $en \Leftrightarrow de$, $en \Leftrightarrow fr$, $en \Leftrightarrow es$ (we opted for an english-centric setting). 
- **Tokenizer training set:** [CC100](https://huggingface.co/datasets/cc100).
- **Training set:** [CCMatrix](https://huggingface.co/datasets/yhavinga/ccmatrix).
- **Validation set:**
  - _newstest2012_ for $en \Leftrightarrow es$.
  - _newstest2013_ for $en \Leftrightarrow de$ and $en \Leftrightarrow fr$.
- **Test set:**
  - _newstest2013_ for $en \Leftrightarrow es$.
  - _newstest2014_ for $en \Leftrightarrow de$ and $en \Leftrightarrow fr$.

The validation and test sets are in a [personal public repository](https://huggingface.co/datasets/thesistranslation/wmt14) on the Huggingface hub.

---

## :robot: Models
The NAT models' names are taken from the following [survey](https://arxiv.org/pdf/2204.09269.pdf).
### AR (autoregressive) models
- [Transformer](https://arxiv.org/abs/1706.03762)

### NAR (non-autoregressive) models
- [FT-NAT](https://arxiv.org/abs/1711.02281) (_WIP_)
- [CMLM](https://arxiv.org/abs/1904.09324)
- [GLAT](https://arxiv.org/abs/2008.07905) (_WIP_)
