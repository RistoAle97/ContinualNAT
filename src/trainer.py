import os
import yaml
from torch.utils.data import DataLoader, ConcatDataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from datasets import load_dataset
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from src.data import *
from src.models import TransformerCore, CMLM
from src.utils import MBART_LANG_MAP, compute_accumulation_steps
from typing import Dict, List, Set, Tuple, Union, NewType

DatasetList = NewType("DatasetList", List[Union[TranslationDataset, IterableTranslationDataset]])


class MultilingualTrainer:

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: Union[int, None] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 use_cls_token: bool = False,
                 train_dataset_cache_dir: str = None,
                 val_dataset_cache_dir: str = None,
                 nmt_directions: List[str] = None,
                 train_bsz: int = 128,
                 val_bsz: int = 128,
                 tokens_per_batch: int = None,
                 train_steps: int = 100000,
                 val_every_n_steps: int = 10000,
                 log_every_n_steps: int = 500,
                 ckpt_every_n_steps: int = 10000,
                 streaming: bool = False) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.use_cls_token = use_cls_token
        self.nmt_directions, self.lang_pairs = self.__build_translation_directions(nmt_directions)
        self.train_bsz = train_bsz
        self.val_bsz = val_bsz
        self.tokens_per_batch = tokens_per_batch if tokens_per_batch is not None else train_bsz * max_length
        self.train_steps = train_steps
        self.val_every_n_steps = val_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.ckpt_every_n_steps = ckpt_every_n_steps
        self.accumulation_steps = compute_accumulation_steps(self.train_bsz, self.max_length, self.tokens_per_batch)
        self.streaming = streaming
        self.train_datasets, self.val_datasets = self.__build_datasets(train_dataset_cache_dir, val_dataset_cache_dir)

    @staticmethod
    def __build_translation_directions(nmt_directions: List[str]) -> Tuple[Set[str], Set[Tuple[str, str]]]:
        directions = set()  # all the translation directions
        lang_pairs = set()  # unique language pairs (en-es and es-en is considered as one pair)
        for direction in nmt_directions:
            langs = direction.split("-")
            if len(langs) != 2:
                raise ValueError("The language pairs must be between two languages and in the format"
                                 "\"src_lang\"-\"tgt_lang\" (e.g.: es-en).")

            src_lang, tgt_lang = langs[0], langs[-1]
            supported_langs = list(MBART_LANG_MAP.keys())
            if src_lang not in supported_langs or tgt_lang not in supported_langs:
                raise ValueError("Both source and target languages codes should follow the ISO format.")

            directions.add(direction)
            lang_pair = (src_lang, tgt_lang) if src_lang < tgt_lang else (tgt_lang, src_lang)
            if lang_pair not in lang_pairs:
                lang_pairs.add(lang_pair)

        return directions, lang_pairs

    def __build_datasets(self, train_cache_dir: str, val_cache_dir: str) -> Tuple[DatasetList, DatasetList]:
        datasets_train = {}
        datasets_val = {}
        translation_train_datasets = DatasetList([])
        translation_val_datasets = DatasetList([])

        # Compute the number of sentences per language pair
        '''sentences_to_consider = self.train_bsz * self.train_steps * self.accumulation_steps
        sentences_per_lang_pair = [sentences_to_consider // len(self.nmt_directions)] * len(self.lang_pairs)
        remaining_sentences = sentences_to_consider % len(self.lang_pairs)
        if remaining_sentences > 0:
            for i in range(remaining_sentences):
                sentences_per_lang_pair[i] += 1'''

        with open("duplicates.yaml") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)

        # Load the datasets and build a translation dataset for each translation direction
        lang_pairs_seen = []
        for direction in self.nmt_directions:
            src_lang, tgt_lang = direction.split("-")
            lang_pair = (src_lang, tgt_lang) if src_lang < tgt_lang else (tgt_lang, src_lang)
            idxs_to_skip = set(config[f"duplicates_{src_lang}_{tgt_lang}"] +
                               config[f"duplicates_{tgt_lang}_{src_lang}"])
            if lang_pair not in lang_pairs_seen:
                first_lang, second_lang = lang_pair

                # Load the train and validation datasets
                '''dataset_train = load_dataset("yhavinga/ccmatrix", f"{first_lang}-{second_lang}",
                                             split=f"train[:30000000]",
                                             cache_dir=train_cache_dir, verification_mode="no_checks")'''
                if first_lang == "en":
                    first_lang = second_lang
                    second_lang = "en"

                dataset_train = load_dataset("wmt14", f"{first_lang}-{second_lang}", split="train",
                                             cache_dir=val_cache_dir, verification_mode="no_checks")

                dataset_val = load_dataset("wmt14", f"{first_lang}-{second_lang}", split="validation",
                                           cache_dir=val_cache_dir, verification_mode="no_checks")
                '''dataset_val = concatenate_datasets([dataset_val["dev"], dataset_val["devtest"]])
                dataset_val_dict = []
                for sentence_pair in dataset_val:
                    dataset_val_dict.append({src_lang: sentence_pair[f"sentence_{first_lang}"],
                                             tgt_lang: sentence_pair[f"sentence_{second_lang}"]})

                dataset_val = dataset_val.add_column("translation", dataset_val_dict)
                dataset_val = dataset_val.remove_columns([f"sentence_{first_lang}", f"sentence_{second_lang}"])'''
                lang_pairs_seen.append(lang_pair)
                datasets_train[lang_pair] = dataset_train
                datasets_val[lang_pair] = dataset_val
            else:
                dataset_train = datasets_train[lang_pair]
                dataset_val = datasets_val[lang_pair]

            dataset_train = TranslationDataset(src_lang, tgt_lang, dataset_train, self.tokenizer,
                                               max_length=self.max_length, padding=self.padding,
                                               use_cls_token=self.use_cls_token, skip_idxs=idxs_to_skip)
            dataset_val = TranslationDataset(src_lang, tgt_lang, dataset_val, self.tokenizer,
                                             max_length=self.max_length, padding=self.padding,
                                             use_cls_token=self.use_cls_token)
            translation_train_datasets.append(dataset_train)
            translation_val_datasets.append(dataset_val)

        return translation_train_datasets, translation_val_datasets

    def __build_dataloaders(self, model: TransformerCore) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        # Train dataloader
        train_dataset = ConcatDataset(self.train_datasets)
        if isinstance(model, CMLM):
            batch_collator_train = BatchCollatorCMLM(self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, True)
        else:
            batch_collator_train = BatchCollator(shift_lang_token=True, pad_token_id=self.tokenizer.pad_token_id)

        train_dataloader = DataLoader(train_dataset, batch_size=self.train_bsz, shuffle=True, num_workers=8,
                                      collate_fn=batch_collator_train, pin_memory=True)

        # Validation dataloaders (one for each translation direction)
        val_dataloaders = {}
        for dataset in self.val_datasets:
            src_lang = dataset.src_lang
            tgt_lang = dataset.tgt_lang
            if isinstance(model, CMLM):
                batch_collator_val = BatchCollatorCMLM(self.tokenizer.pad_token_id, self.tokenizer.mask_token_id)
            else:
                batch_collator_val = BatchCollator(shift_lang_token=True, pad_token_id=self.tokenizer.pad_token_id)

            val_dataloaders[f"{src_lang}_{tgt_lang}"] = DataLoader(dataset, self.val_bsz, num_workers=8,
                                                                   collate_fn=batch_collator_val, pin_memory=True)

        return train_dataloader, val_dataloaders

    def __compute_logger_version(self, model: TransformerCore) -> str:
        logger_version = f"{model.__class__.__name__}"
        if len(self.lang_pairs) > 1:
            logger_version += "_multilingual"
        else:
            first_lang, second_lang = list(self.nmt_directions)[0].split("-")
            logger_version += f"_{first_lang}_{second_lang}"
            if len(self.nmt_directions) > 1:
                logger_version += "_both"

        v_num = 0
        while os.path.exists(f"logs/{logger_version}_{v_num}"):
            v_num += 1

        logger_version += f"_{v_num}"
        return logger_version

    def train(self, model: TransformerCore) -> None:
        # Dataloaders
        train_dataloader, val_dataloaders = self.__build_dataloaders(model)

        # Logger and other train callbacks
        logger_version = self.__compute_logger_version(model)
        logger = TensorBoardLogger("", name="logs", version=logger_version)
        checkpoint = ModelCheckpoint(save_top_k=2, monitor="mean_BLEU", mode="max",
                                     every_n_train_steps=self.ckpt_every_n_steps)
        prog_bar_theme = RichProgressBarTheme(description="red", progress_bar="dark_blue",
                                              progress_bar_finished="dark_blue", progress_bar_pulse="dark_blue")
        prog_bar = RichProgressBar(theme=prog_bar_theme)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer_callbacks = [checkpoint, lr_monitor, prog_bar]

        # Train the model
        trainer = Trainer(devices=1, precision="16-mixed", logger=logger, max_steps=self.train_steps,
                          val_check_interval=self.val_every_n_steps * self.accumulation_steps,
                          log_every_n_steps=self.log_every_n_steps, accumulate_grad_batches=self.accumulation_steps,
                          gradient_clip_val=1.0, callbacks=trainer_callbacks)
        trainer.fit(model, train_dataloader, val_dataloaders)
