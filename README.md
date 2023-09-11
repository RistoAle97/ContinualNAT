<div align="center">

# ContinualNAT
<img src="assets/unipiLogo.png" alt="Unipi Logo" width="230"/><br>

**M.Sc. thesis on Continual Learning for multilingual non-autoregressive Neural Machine Translation (NAT).**

---

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)]()

[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)
[![Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white)](https://github.com/Lightning-AI/lightning)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-huggingface-FF9D00?style=for-the-badge&labelColor=FFD21E)](https://github.com/huggingface/transformers)

---
</div>

## :pushpin: Abstract
The Transformer architecture changed the world of Natural Language Processing and Deep Learning in general by setting new state-of-the-art scores for many fields and, nowadays, it is the go-to solution when approaching a new problem, but it comes with a limitation: its inference speed. The Transformer uses the parallelizable mechanism of self-attention during training in order to avoid the typical recurrence of RNN, but the use of an autoregressive (AR) decoder limits its full potential at inference time: at each time-step, only one token is generated.

In order to reach the full potential of the Transformer architecture a new kind of non-autoregressive (NAR) models were introduced, but it turned out that their performances were (and still are) way behind their AR counterparts.

Our purpose is to investigate the goodness of one of the most famous NAR models in multilingual Neural Machine Translation (NMT) setting, while also testing its behaviour under a simple Continual Learning approach.

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
- **Translation directions:** $en \Leftrightarrow \lbrace de, fr, es \rbrace$.
- **Tokenizer training set:** [CC100](https://huggingface.co/datasets/cc100).
- **Training set:** a distilled version of [CCMatrix](https://huggingface.co/datasets/yhavinga/ccmatrix), where only the first 30m of sentence pairs are considered.
- **Validation set:**
  - _newstest2012_ for $en \Leftrightarrow es$.
  - _newstest2013_ for $en \Leftrightarrow de$ and $en \Leftrightarrow fr$.
- **Test set:**
  - _newstest2013_ for $en \Leftrightarrow es$.
  - _newstest2014_ for $en \Leftrightarrow de$ and $en \Leftrightarrow fr$.

The validation and test sets are in a [personal public repository](https://huggingface.co/datasets/thesistranslation/wmt14) on the Huggingface hub.

---

## :hammer_and_wrench: Distillation
We employed the so-called [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) to translate the first 30m of sentences in the target language from CCMatrix using an autoregressive teacher model. Such translations are then used as the references when training the models.

First, we converted the teacher models into [CTranslate2](https://github.com/OpenNMT/CTranslate2) format with
```bash
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir ct2-opus-mt-en-de
```
then, we used the following command to distill the datasets and upload them on the Huggingface hub
```bash
python distill_ccmatrix.py --src en --tgt de
```
Hereafter are all the teacher models and distilled datasets used for our experiments.

| Translation Direction |                           Teacher Model                            |                                           Distilled Dataset                                            |
|:---------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|
| $en \Rightarrow de $  | [opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) | [distilled-ccmatrix-en-de](https://huggingface.co/datasets/thesistranslation/distilled-ccmatrix-en-de) |
| $de \Rightarrow en $  | [opus-mt-de-en](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) | [distilled-ccmatrix-de-en](https://huggingface.co/datasets/thesistranslation/distilled-ccmatrix-de-en) |
| $en \Rightarrow fr $  | [opus-mt-en-fr](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) | [distilled-ccmatrix-en-fr](https://huggingface.co/datasets/thesistranslation/distilled-ccmatrix-en-fr) |
| $fr \Rightarrow en $  | [opus-mt-fr-en](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en) | [distilled-ccmatrix-fr-en](https://huggingface.co/datasets/thesistranslation/distilled-ccmatrix-fr-en) |
| $en \Rightarrow es $  | [opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) | [distilled-ccmatrix-en-es](https://huggingface.co/datasets/thesistranslation/distilled-ccmatrix-en-es) |
| $es \Rightarrow en $  | [opus-mt-es-en](https://huggingface.co/Helsinki-NLP/opus-mt-es-en) | [distilled-ccmatrix-es-en](https://huggingface.co/datasets/thesistranslation/distilled-ccmatrix-es-en) |

---

## :label: Continual setting
The models were trained on three subsequent experiences each made up of two translation directions involving a language pair. At the end of each experience, a fixed-size buffer is filled with random samples following a simple reservoir sampling approach.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/continual_setting_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/continual_setting_light.svg">
  <img alt="Continual setting" src="assets/continual_setting_dark.svg">
</picture>
</div>

---

## :robot: Models
The NAT models' names are taken from the following [survey](https://arxiv.org/pdf/2204.09269.pdf).
### AR (autoregressive) models
- [Transformer](https://arxiv.org/abs/1706.03762)

### NAR (non-autoregressive) models

###### Semi-NAT
- [CMLM](https://arxiv.org/abs/1904.09324)

###### Fully-NAT
- CMLM with [GLAT](https://arxiv.org/abs/2008.07905) training

### Some small useful tutorials using trained models
<details>
<summary>Evaluation</summary>

```python
import torch
from datasets import load_dataset
from transformers import MBartTokenizerFast

from continualnat.data import TranslationDataset
from continualnat.metrics import compute_sacrebleu
from continualnat.models.cmlm import CMLMConfig, CMLM

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tokenizer and some useful tokens
tokenizer = MBartTokenizerFast(tokenizer_file="tokenizers/sp_32k.json", model_max_length=1024,
                               cls_token="<length>")
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
mask_token_id = tokenizer.mask_token_id

# Load the dataset
wmt_en_de = load_dataset("thesistranslation/wmt14", "de-en",
                         cache_dir="/disk1/a.ristori/datasets/wmt14",
                         verification_mode="no_checks")
wmt_en_de_test = TranslationDataset("en", "de", wmt_en_de["test"], tokenizer, max_length=128)

# Load the model
model_state_dict = torch.load("path/to/your/saved/model")
model_config = CMLMConfig(len(tokenizer), bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                          pad_token_id=pad_token_id, mask_token_id=mask_token_id, length_token_id=None,
                          pooler_size=256, glat_training=True)
model = CMLM(model_config)
model.load_state_dict(model_state_dict)
model.to(device)

# Compute BLEU score
bleu_scores = compute_sacrebleu(model, wmt_en_de_test, tokenizer, metric_tokenize={"13a", "intl"})
print(bleu_scores)
```

```
{'intl': 22.757592245926443, '13a': 22.19058951758056}
```

</details>
<details>
<summary>Visualization of mask-predict steps</summary>

```python
import torch
from transformers import MBartTokenizerFast
from continualnat.models.cmlm import CMLMConfig, CMLM, tabulate_mask_predict_steps

# Tokenizer and some useful tokens
tokenizer = MBartTokenizerFast(tokenizer_file="tokenizers/sp_32k.json", model_max_length=1024,
                               cls_token="<length>")
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
mask_token_id = tokenizer.mask_token_id

# Load the model
model_state_dict = torch.load("path/to/your/saved/model")
model_config = CMLMConfig(len(tokenizer), bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                          pad_token_id=pad_token_id, mask_token_id=mask_token_id, length_token_id=None,
                          pooler_size=256, glat_training=True)
model = CMLM(model_config)
model.load_state_dict(model_state_dict)

# Translate the sentences
src_sentences = ["What are you doing for the session?", "That was amazing, how did you do it?"]
tokenized_sentences = tokenizer(src_sentences, truncation=True, padding="longest", return_tensors="pt")
output = model.generate(tokenized_sentences.input_ids, tokenizer.lang_code_to_id["de_DE"])
translations_tokens, tokens_ids_at_each_step = output

# Tabulate the tokens generated at each step by mask-predict
tabulated_tokens_at_each_step, _ = tabulate_mask_predict_steps(tokens_ids_at_each_step, tokenizer)

# Let's show the mask-predict steps for the first sentence
print(tabulated_tokens_at_each_step[0])
```

```
-  ------  --------  ------  ------  ------  ----------  ------  ----  -----
0  <mask>  <mask>    <mask>  <mask>  <mask>  <mask>      <mask>  </s>  de_DE
1  ▁Was    ▁machen   ▁Sie    ▁für    ▁die    ▁Sitzung   ?       </s>  de_DE
-  ------  --------  ------  ------  ------  ----------  ------  ----  -----
```

</details>

---

## :memo: License
This project is [MIT licensed](https://github.com/RistoAle97/ContinualNAT/blob/main/LICENSE).
