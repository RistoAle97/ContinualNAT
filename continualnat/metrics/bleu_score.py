from typing import Dict, Set, Union

import datasets
import evaluate
import sacrebleu
import torch
from ctranslate2 import Translator
from torch.utils.data import DataLoader
from transformers import MBartTokenizer, MBartTokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm.auto import tqdm

from continualnat.data.batch_collators import BatchCollator
from continualnat.data.datasets import TranslationDataset
from continualnat.models.cmlm.cmlm import CMLM
from continualnat.models.core.transformer_core import TransformerCore
from continualnat.models.glat.glat import GLAT

TOKENIZERS = set(tokenizer for tokenizer in sacrebleu.BLEU.TOKENIZERS)


def compute_sacrebleu(
    model: TransformerCore,
    dataset: TranslationDataset,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    bsz: int = 32,
    iterations: int = None,
    prog_bar: bool = True,
    metric_tokenize: Set[str] = None,
) -> Dict[str, float]:
    """
    Computes the SacreBLEU score of a model built with the continualnat package for a given dataset.
    :param model: the model built with this library.
    :param dataset: the TranslationDataset on which the SacreBLEU score will be computed.
    :param tokenizer: the tokenizer used by the model, if no tokenizer is passed, then the dataset's one will be
        used instead (default=None).
    :param bsz: the batch size (default=32).
    :param iterations: the number of iterations of mask-predict, it's only applied if the model is CMLM (default=None).
    :param prog_bar: whether to show the progress bar (default=True).
    :param metric_tokenize: the tokenizers used by the SacreBLEU computation (default=None).
    :return: a Dict containing the SacreBLEU score for each tokenizer.
    """
    if metric_tokenize is None:
        metric_tokenize = {"none"}

    if metric_tokenize.intersection(TOKENIZERS) != metric_tokenize:
        raise ValueError(
            "Wrong tokenizer passed for the SacreBLEU metric, use one or more of the following: \"none\", \"zh\","
            "\"13a\", \"char\", \"ja-mecab\"."
        )

    if tokenizer is None:
        tokenizer = dataset.tokenizer

    scb = evaluate.load("sacrebleu")
    device = model.device
    batch_collator_args = {
        "is_mlm": False,
        "shift_lang_token": isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)),
        "return_lengths": False,
        "pad_token_id": model.pad_token_id,
        "mask_token_id": model.mask_token_id if hasattr(model, "mask_token_id") else tokenizer.mask_token_id,
        "p_masking": 0.0,
    }
    generation_kwargs = {}
    if isinstance(model, (CMLM, GLAT)):
        batch_collator_args["is_mlm"] = True
        batch_collator_args["shift_lang_token"] = False
        batch_collator_args["return_lengths"] = True
        if isinstance(model, CMLM):
            batch_collator_args["p_masking"] = 1.0
            generation_kwargs = {"iterations": iterations}
        else:
            batch_collator_args["p_masking"] = 0.0

    batch_collator = BatchCollator(**batch_collator_args)
    dataloader = DataLoader(dataset, batch_size=bsz, collate_fn=batch_collator)
    translations = []
    targets = []
    dataloader = tqdm(dataloader) if prog_bar else dataloader
    for batch in dataloader:
        input_ids = batch["input_ids"]
        tgt_lang_code_id = tokenizer.lang_code_to_id[dataset.tgt_lang_code]
        translation = model.generate(input_ids.to(device), tgt_lang_code_id, **generation_kwargs)
        if isinstance(model, CMLM):
            translation, _ = translation

        # translation, _ = translation if isinstance(model, CMLM) else translation
        decoded_translation = tokenizer.batch_decode(translation, skip_special_tokens=True)
        translations.extend(decoded_translation)
        targets.extend(batch["references"])

    bleu_scores = {tokenize: 0 for tokenize in metric_tokenize}
    for tokenize in metric_tokenize:
        bleu_scores[tokenize] = scb.compute(predictions=translations, references=targets, tokenize=tokenize)["score"]

    return bleu_scores


def compute_sacrebleu_ct2(
    model: str,
    dataset: datasets.Dataset,
    src_lang: str,
    tgt_lang: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    beam_size: int,
    device: torch.device,
    bsz: int = 512,
    prog_bar: bool = True,
    metric_tokenize: Set[str] = None,
) -> Dict[str, float]:
    """
    Computes the SacreBLEU score of a CTranslate2 model (only for the ones converted from a Hugginface's Transformers
    model) for a given dataset.
    :param model: the CTranslate model converted from a Transformers model, a path to the direcotry containing the model
        should be passed.
    :param dataset: the torch.Dataset on which the SacreBLEU score will be computed.
    :param src_lang: the source language.
    :param tgt_lang: the target language.
    :param tokenizer: the tokenizer used by the model.
    :param beam_size: the number of beams to keep during beam decoding.
    :param device: the device used by the computation.
    :param bsz: the batch size, it is higly recommended to use the highest possible value for the available machine
        (default=512).
    :param prog_bar: whether to show the progress bar (default=True)
    :param metric_tokenize: the tokenizers used by the SacreBLEU computation (default=None).
    :return: a Dict containing the SacreBLEU score for each tokenizer.
    """
    if metric_tokenize is None:
        metric_tokenize = {"none"}

    if metric_tokenize.intersection(TOKENIZERS) != metric_tokenize:
        raise ValueError(
            "Wrong tokenizer passed for the SacreBLEU metric, use one or more of the following: \"none\", \"zh\","
            "\"13a\", \"char\", \"ja-mecab\"."
        )

    scb = evaluate.load("sacrebleu")
    max_length = tokenizer.model_max_length
    translator = Translator(model, device=device.type, device_index=device.index)
    dataloader = DataLoader(dataset, batch_size=bsz)
    translations = []
    targets = []
    dataloader = tqdm(dataloader) if prog_bar else dataloader
    for batch in dataloader:
        src_sentences = batch["translation"][src_lang]
        src_tokens = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(src_sentence, truncation=True, max_length=max_length))
            for src_sentence in src_sentences
        ]
        generated_translations = translator.translate_batch(
            src_tokens, beam_size=beam_size, max_decoding_length=max_length
        )
        translations_tokens = [translation.hypotheses[0] for translation in generated_translations]
        translations_ids = [tokenizer.convert_tokens_to_ids(tgt_tokens) for tgt_tokens in translations_tokens]
        decoded_translation = tokenizer.batch_decode(translations_ids, skip_special_tokens=True)
        translations.extend(decoded_translation)
        targets.extend([tgt_sentence] for tgt_sentence in batch["translation"][tgt_lang])

    bleu_scores = {tokenize: 0 for tokenize in metric_tokenize}
    for tokenize in metric_tokenize:
        bleu_scores[tokenize] = scb.compute(predictions=translations, references=targets, tokenize=tokenize)["score"]

    return bleu_scores
