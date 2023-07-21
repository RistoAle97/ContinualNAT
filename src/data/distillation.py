from typing import Union

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.models.core.transformer_core import TransformerCore


def distill_dataset(teacher: Union[PreTrainedModel, TransformerCore],
                    tokenizer: PreTrainedTokenizer,
                    dataset: Dataset,
                    src_lang: str,
                    tgt_lang: str,
                    bsz: int = 32,
                    path: str = None,
                    prog_bar: bool = True) -> None:
    """
    Apply sequence-level knowledge distillation on a Hugginface dataset.
    :param teacher: the model used to distill the dataset.
    :param tokenizer: the tokenizer used by the model.
    :param dataset: the Hugginface dataset.
    :param src_lang: the source language.
    :param tgt_lang: the target language.
    :param bsz: batch size used during the distillation (dfefault=32).
    :param path: where to save the distilled dataset, if None then the distilled dataset will be saved inside the
        current working directory with the name "distilled_buildername.source_target.target" (default=None).
    :param prog_bar: whether to show the progrees bar during the distillation (default=True).
    """
    device = teacher.device
    dataloader = DataLoader(dataset, batch_size=bsz, pin_memory=True)
    dataloader = tqdm(dataloader) if prog_bar else dataloader
    distill_path = f"distilled_{dataset.builder_name}.{src_lang}_{tgt_lang}.{tgt_lang}" if path is None else path
    with open(distill_path, "w") as datafile:
        for batch in dataloader:
            sentences_to_distill = batch["translation"][src_lang]
            if tgt_lang == "de":
                # There are some sentences inside the WMT14 en-de dataset with the wrong quotation mark
                sentences_to_distill = [sentence.replace("\u201d", "\"") for sentence in sentences_to_distill]

            input_ids = tokenizer(sentences_to_distill, truncation=True, max_length=tokenizer.model_max_length,
                                  padding="longest", return_tensors="pt")["input_ids"]
            translation = teacher.generate(input_ids.to(device))
            decoded_translation = tokenizer.batch_decode(translation, skip_special_tokens=True)
            decoded_translation = [distilled_translation + "\n" for distilled_translation in decoded_translation]
            datafile.writelines(decoded_translation)
