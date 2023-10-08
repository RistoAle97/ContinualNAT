# Tokenizers
All the tokenizers were trained on an equal number of English, French, German and Spanish sentences from the [CC100 dataset](https://huggingface.co/datasets/cc100) and are based on [Sentencepiece](https://arxiv.org/abs/1808.06226.pdf) with [BPE](https://arxiv.org/abs/1508.07909). Here you will find their vocabularies in a json format.

You can load them by following this simple script (keep in  mind that they only work with the fast version of the Hugginface tokenizers)
```python
from transformers import MBartTokenizerFast

tokenizer = MBartTokenizerFast(
    tokenizer_file="tokenizers/sp_32k.json",
    model_max_length=1024,
    cls_token="<length>",
)
```

See also the [notebook](https://github.com/RistoAle97/ContinualNAT/blob/mask_predict/notebooks/train_tokenizer.ipynb) on how to train one of them.