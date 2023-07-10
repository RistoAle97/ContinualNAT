import os
from torch.utils.data import DataLoader, ConcatDataset
from transformers import PreTrainedTokenizerBase
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from src.data import TranslationDataset, BatchCollator, BatchCollatorCMLM
from src.models import TransformerCore, CMLM
from src.utils import MBART_LANG_MAP, compute_accumulation_steps
from typing import Dict, List, Set, Tuple


class MultilingualTrainer:

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 train_steps: int = 100000,
                 val_every_n_steps: int = 10000,
                 log_every_n_steps: int = 500,
                 ckpt_every_n_steps: int = 10000,
                 log_directory: str = "",
                 use_wandb: bool = True) -> None:
        self.tokenizer = tokenizer
        self.train_steps = train_steps
        self.val_every_n_steps = val_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.ckpt_every_n_steps = ckpt_every_n_steps
        self.log_directory = log_directory
        self.use_wandb = use_wandb

    @staticmethod
    def __build_nmt_directions(train_datasets: List[TranslationDataset]) -> Tuple[Set[str], Set[Tuple[str, str]]]:
        lang_pairs = set()  # unique language pairs (e.g.: en-es and es-en is considered as a single pair)
        train_directions = set([f"{dataset.src_lang}-{dataset.tgt_lang}" for dataset in train_datasets])
        for direction in train_directions:
            langs = direction.split("-")
            src_lang, tgt_lang = langs[0], langs[-1]
            supported_langs = list(MBART_LANG_MAP.keys())
            if src_lang not in supported_langs or tgt_lang not in supported_langs:
                raise ValueError("Both source and target languages codes should follow the ISO format.")

            lang_pair = (src_lang, tgt_lang) if src_lang < tgt_lang else (tgt_lang, src_lang)
            if lang_pair not in lang_pairs:
                lang_pairs.add(lang_pair)

        return train_directions, lang_pairs

    def __build_dataloaders(self,
                            model: TransformerCore,
                            train_datasets: List[TranslationDataset],
                            val_datasets: List[TranslationDataset],
                            train_bsz: int = 128,
                            val_bsz: int = 128) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        # Train dataloader
        train_dataset = ConcatDataset(train_datasets)
        if isinstance(model, CMLM):
            batch_collator_train = BatchCollatorCMLM(self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, True)
        else:
            batch_collator_train = BatchCollator(shift_lang_token=True, pad_token_id=self.tokenizer.pad_token_id)

        train_dataloader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True, num_workers=8,
                                      collate_fn=batch_collator_train, pin_memory=True)

        # Validation dataloaders (one for each translation direction)
        val_dataloaders = {}
        for dataset in val_datasets:
            src_lang = dataset.src_lang
            tgt_lang = dataset.tgt_lang
            if isinstance(model, CMLM):
                batch_collator_val = BatchCollatorCMLM(self.tokenizer.pad_token_id, self.tokenizer.mask_token_id)
            else:
                batch_collator_val = BatchCollator(shift_lang_token=True, pad_token_id=self.tokenizer.pad_token_id)

            val_dataloaders[f"{src_lang}_{tgt_lang}"] = DataLoader(dataset, val_bsz, num_workers=8,
                                                                   collate_fn=batch_collator_val, pin_memory=True)

        return train_dataloader, val_dataloaders

    @staticmethod
    def __compute_logger_version(model: TransformerCore,
                                 nmt_directions: Set[str],
                                 lang_pairs: Set[Tuple[str, str]]) -> str:
        logger_version = f"{model.__class__.__name__}"
        if len(lang_pairs) > 1:
            logger_version += "_multilingual"
        else:
            first_lang, second_lang = list(nmt_directions)[0].split("-")
            logger_version += f"_{first_lang}_{second_lang}"
            if len(nmt_directions) > 1:
                logger_version += "_both"

        v_num = 0
        while os.path.exists(f"logs/{logger_version}_{v_num}"):
            v_num += 1

        logger_version += f"_{v_num}"
        return logger_version

    def train(self,
              model: TransformerCore,
              train_datasets: List[TranslationDataset],
              val_datasets: List[TranslationDataset],
              train_bsz: int = 128,
              val_bsz: int = 128,
              tokens_per_batch: int = None,
              logger_version: str = None) -> None:
        # Build translation directions and unique lang pairs
        nmt_directions, lang_pairs = self.__build_nmt_directions(train_datasets)

        # Dataloaders
        train_dataloader, val_dataloaders = self.__build_dataloaders(model, train_datasets, val_datasets,
                                                                     train_bsz, val_bsz)

        # Accumulation steps
        max_length = train_datasets[0].tokenizer_state["max_length"]
        tokens_per_batch = tokens_per_batch if tokens_per_batch is not None else train_bsz * max_length
        accumulation_steps = compute_accumulation_steps(train_bsz, max_length, tokens_per_batch)

        # Logger and other train callbacks
        if logger_version is None:
            logger_version = self.__compute_logger_version(model, nmt_directions, lang_pairs)

        if self.use_wandb:
            logger = WandbLogger(self.log_directory, name="logs", project=logger_version)
        else:
            logger = TensorBoardLogger(self.log_directory, name="logs", version=logger_version)

        checkpoint = ModelCheckpoint(save_top_k=2, monitor="mean_BLEU", mode="max",
                                     every_n_train_steps=self.ckpt_every_n_steps)
        prog_bar_theme = RichProgressBarTheme(description="red", progress_bar="dark_blue",
                                              progress_bar_finished="dark_blue", progress_bar_pulse="dark_blue")
        prog_bar = RichProgressBar(theme=prog_bar_theme)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer_callbacks = [checkpoint, lr_monitor, prog_bar]

        # Train the model
        trainer = Trainer(devices=1, precision="16-mixed", logger=logger, max_steps=self.train_steps,
                          val_check_interval=self.val_every_n_steps * accumulation_steps,
                          check_val_every_n_epoch=None, log_every_n_steps=self.log_every_n_steps,
                          accumulate_grad_batches=accumulation_steps, gradient_clip_val=1.0,
                          callbacks=trainer_callbacks)
        trainer.fit(model, train_dataloader, val_dataloaders)
