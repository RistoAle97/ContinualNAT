from typing import Dict, Set

import evaluate
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, MBartTokenizer, MBartTokenizerFast
from tqdm.auto import tqdm

from src.data.datasets import TranslationDataset
from src.data.collators import BatchCollator, BatchCollatorCMLM
from src.models.cmlm.cmlm import CMLM
from src.models.core.transformer_core import TransformerCore


def compute_sacrebleu(model: TransformerCore,
                      dataset: TranslationDataset,
                      tokenizer: PreTrainedTokenizerBase,
                      bsz: int = 32,
                      prog_bar: bool = False,
                      metric_tokenize: Set[str] = None) -> Dict[str, float]:
    if metric_tokenize is None:
        metric_tokenize = {"none"}

    if metric_tokenize.intersection({"none", "zh", "13a", "intl", "char", "ja-mecab"}) != metric_tokenize:
        raise ValueError("Wrong tokenizer passed for the SacreBLEU metric, use one or more of the following: \"none\","
                         "\"zh\", \"13a\", \"char\", \"ja-mecab\".")

    scb = evaluate.load("sacrebleu")
    device = model.device
    if isinstance(model, CMLM):
        batch_collator = BatchCollatorCMLM(model.pad_token_id, model.mask_token_id)
    else:
        shift_lang_token = True if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)) else False
        batch_collator = BatchCollator(shift_lang_token=shift_lang_token, pad_token_id=model.pad_token_id)

    dataloader = DataLoader(dataset, batch_size=bsz, collate_fn=batch_collator)
    translations = []
    targets = []
    dataloader = tqdm(dataloader) if prog_bar else dataloader
    for i, batch in enumerate(dataloader):
        translation = model.generate(batch["input_ids"].to(device), tokenizer.lang_code_to_id[dataset.tgt_lang_code])
        decoded_translation = tokenizer.batch_decode(translation, skip_special_tokens=True)
        translations.extend(decoded_translation)
        targets.extend(batch["references"])

    bleu_scores = {tokenize: 0 for tokenize in metric_tokenize}
    for tokenize in metric_tokenize:
        bleu_scores[tokenize] = scb.compute(predictions=translations, references=targets, tokenize=tokenize)["score"]

    return bleu_scores
