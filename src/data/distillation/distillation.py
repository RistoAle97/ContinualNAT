import os
import tarfile
from typing import Union

import datasets
import torch
from ctranslate2 import Translator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.models.core.transformer_core import TransformerCore


def distill_dataset(teacher: Union[PreTrainedModel, TransformerCore, str],
                    tokenizer: PreTrainedTokenizer,
                    dataset: datasets.Dataset,
                    dataset_name: str,
                    src_lang: str,
                    tgt_lang: str,
                    device: torch.device,
                    beam_size: int,
                    bsz: int = 32,
                    save_dir: str = None,
                    prog_bar: bool = True) -> None:
    """
    Apply sequence-level knowledge distillation on a Hugginface dataset.
    :param teacher: the model used to distill the dataset. If a string is passed, then a converted ctranslate model
        will be used. Take a look at the following guide on how to convert a Huggingface's transformers pre-trained
        model to CTranslate format: https://opennmt.net/CTranslate2/guides/transformers.html.
    :param tokenizer: the tokenizer used by the model.
    :param dataset: the Hugginface dataset.
    :param dataset_name: the dataset's name that will be used to save it.
    :param src_lang: the source language.
    :param tgt_lang: the target language.
    :param device: the device on which to run the distillation.
    :param beam_size: the beam size used by the model during decoding.
    :param bsz: batch size used during the distillation (dfefault=32).
    :param save_dir: the directory in which to save the distilled dataset's files, if None then the files will be saved
        in the current directory (default=None).
    :param prog_bar: whether to show the progrees bar during the distillation (default=True).
    """
    if isinstance(teacher, str):
        translator = Translator(teacher, device=device.type, device_index=device.index)
    else:
        translator = None

    max_length = tokenizer.model_max_length
    dataloader = DataLoader(dataset, batch_size=bsz, pin_memory=True)
    save_dir = "" if save_dir is None else save_dir
    src_distill_path = f"{save_dir}/distilled_{dataset_name}.{src_lang}_{tgt_lang}.{src_lang}"
    tgt_distill_path = f"{save_dir}/distilled_{dataset_name}.{src_lang}_{tgt_lang}.{tgt_lang}"
    dataloader = tqdm(dataloader, desc=f"Distilling {src_lang}->{tgt_lang} dataset") if prog_bar else dataloader
    with open(src_distill_path, "w") as src_datafile, open(tgt_distill_path, "w") as tgt_datafile:
        for batch in dataloader:
            sentences_to_distill = batch["translation"][src_lang]
            if tgt_lang == "de" and translator is not None:
                # There are some sentences inside the ccmatrix en-de dataset causing some issues with the Huggingface
                # generate method while encountering the german quotation marks
                sentences_to_distill = [sentence.replace("\u201d", "\"") for sentence in sentences_to_distill]

            if translator is None:
                # Use the CTranslate2 translator
                input_ids = tokenizer(sentences_to_distill, truncation=True, max_length=max_length, padding="longest",
                                      return_tensors="pt")["input_ids"]
                translation_ids = teacher.generate(input_ids.to(device), max_new_tokens=max_length)
                decoded_translation = tokenizer.batch_decode(translation_ids, skip_special_tokens=True)
            else:
                # Use the generate from Huggingface's Transformers or from this package
                input_ids = tokenizer(sentences_to_distill, truncation=True, max_length=max_length)["input_ids"]
                input_tokens = [tokenizer.convert_ids_to_tokens(src_ids) for src_ids in input_ids]
                generated_translation = translator.translate_batch(input_tokens, beam_size=beam_size,
                                                                   max_decoding_length=max_length)
                translation_tokens = [generated_tokens.hypotheses[0] for generated_tokens in generated_translation]
                translation_ids = [tokenizer.convert_tokens_to_ids(tgt_tokens) for tgt_tokens in translation_tokens]
                decoded_translation = tokenizer.batch_decode(translation_ids, skip_special_tokens=True)

            decoded_translation = [distilled_translation + "\n" for distilled_translation in decoded_translation]
            src_sentences = [src_sentence + "\n" for src_sentence in sentences_to_distill]
            src_datafile.writelines(src_sentences)
            tgt_datafile.writelines(decoded_translation)


def compress_datasets(archive_path: str, src_path: str, tgt_path: str) -> None:
    """
    Compress a distilled dataset into a tar.gz file.
    :param archive_path: the path where the archive is going to be saved.
    :param src_path: the source sentences path.
    :param tgt_path: the target sentenes path.
    """
    with tarfile.open(archive_path, "w:gz") as compressed_files:
        compressed_files.add(src_path)
        compressed_files.add(tgt_path)


def push_distilled_dataset_to_hub(cache_dir: str,
                                  repo_id: str,
                                  src_lang: str,
                                  tgt_lang: str,
                                  path_name: str) -> None:
    """
    Push a distilled dataset to the Huggingface Hub, you need to be logged via the CLI (or any other way) in order to
    use this method.
    :param cache_dir: Directory to read/write data. Defaults to "~/.cache/huggingface/datasets".
    :param repo_id: the repository id.
    :param src_lang: the source language.
    :param tgt_lang: the target language.
    :param path_name: the distilled dataset's path without taking into account the source and target languages.
        As an example, if we have datasets/distilled_dataset.en and datasets/distilled_dataset.es you only need to pass
        datasets/distilled_dataset (default=None).
    """
    path = os.path.dirname(os.path.abspath(__file__))  # this file directory's name
    path = os.path.join(path, "distilled_dataset.py")  # unify the previous path with the dataset builder script
    dataset = load_dataset(path, cache_dir=cache_dir, src_lang=src_lang, tgt_lang=tgt_lang, path_name=path_name)
    dataset.push_to_hub(repo_id)
