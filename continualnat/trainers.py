import os
from typing import Dict, List, Set, Tuple, Union

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler, RandomSampler
from transformers import PreTrainedTokenizerBase

from continualnat.continual.buffer import Buffer
from continualnat.continual.cl_sampler import ExperienceReplaySampler
from continualnat.data.batch_collators import BatchCollator
from continualnat.data.batch_samplers import BatchSamplerCore, HeterogeneousSampler, HomogeneousSampler
from continualnat.data.datasets import TranslationDataset
from continualnat.models.cmlm.cmlm import CMLM
from continualnat.models.core.transformer_core import TransformerCore
from continualnat.utils.utils import MBART_LANG_MAP, compute_accumulation_steps


class TrainerCore:

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 train_steps: int = 100000,
                 val_every_n_steps: int = 10000,
                 log_every_n_steps: int = 500,
                 dataloader_num_workers: int = 8,
                 log_directory: str = "",
                 batch_type: str = "heterogeneous",
                 use_wandb: bool = True) -> None:
        """
        A base for all the custom trainers that wrap the pytorch lightning's trainer.
        :param tokenizer: the tokenizer that will be used by the translation datasets.
        :param train_steps: the number of training updates that the trainer will perform (default=100000).
        :param val_every_n_steps: how often to check the validation set (default=10000).
        :param log_every_n_steps: how often to log the metrics (default=500).
        :param dataloader_num_workers: the number of workers used by both the train and validation dataloaders
            (default=8).
        :param log_directory: the log directory in which to save the logs, "" corresponds to the current working
            directory (default="").
        :param batch_type: the type of batch sampler used by the train dataloader, can be either "heterogeneous" (each
            batch will have one single translation direction) or "homogeneous" (each batch will contain more than
            one translation direction) (default="heterogeneous").
        :param use_wandb: whether to use wandb as a logger, otherwise tensorboard will be used (default=True).
        """
        self.tokenizer = tokenizer
        self.train_steps = train_steps
        self.val_every_n_steps = val_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.num_workers = dataloader_num_workers
        self.log_directory = log_directory
        if batch_type not in ["heterogeneous", "homogeneous"]:
            raise ValueError("The batch type should be \"heterogeneous\" or \"homogeneous\"")

        self.batch_type = batch_type
        self.use_wandb = use_wandb

    def _build_dataloaders_base_(self,
                                 model: TransformerCore,
                                 train_dataset: ConcatDataset[TranslationDataset],
                                 train_batch_sampler: Union[BatchSamplerCore, ExperienceReplaySampler],
                                 val_datasets: List[TranslationDataset],
                                 val_bsz: int = 1280) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        if isinstance(model, CMLM):
            is_mlm = True
            shift_lang_token = False
            return_lengths = True
            p_masking = "random" if not model.glat_training else 1.0
        else:
            is_mlm = False
            shift_lang_token = True
            return_lengths = False
            p_masking = 0.0

        batch_collator_train = BatchCollator(is_mlm=is_mlm, shift_lang_token=shift_lang_token,
                                             return_lengths=return_lengths, pad_token_id=self.tokenizer.pad_token_id,
                                             mask_token_id=self.tokenizer.mask_token_id, p_masking=p_masking)

        train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=self.num_workers,
                                      collate_fn=batch_collator_train, pin_memory=True)

        # Validation dataloaders (one for each translation direction)
        val_dataloaders = {}
        p_masking = 1.0 if isinstance(model, CMLM) else 0.0
        for dataset in val_datasets:
            src_lang = dataset.src_lang
            tgt_lang = dataset.tgt_lang
            batch_collator_val = BatchCollator(is_mlm=is_mlm, shift_lang_token=shift_lang_token,
                                               return_lengths=return_lengths, pad_token_id=self.tokenizer.pad_token_id,
                                               mask_token_id=self.tokenizer.mask_token_id, p_masking=p_masking)

            val_dataloaders[f"{src_lang}_{tgt_lang}"] = DataLoader(dataset, val_bsz, num_workers=self.num_workers,
                                                                   collate_fn=batch_collator_val, pin_memory=True)

        return train_dataloader, val_dataloaders

    def __build_dataloaders(self, *kwargs) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        raise NotImplementedError

    def __compute_logger_version(self, *kwargs) -> str:
        raise NotImplementedError

    def train(self, *kwargs) -> None:
        raise NotImplementedError


class MultilingualTrainer(TrainerCore):

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 train_steps: int = 100000,
                 val_every_n_steps: int = 10000,
                 log_every_n_steps: int = 500,
                 dataloader_num_workers: int = 8,
                 log_directory: str = "",
                 batch_type: str = "heterogeneous",
                 use_wandb: bool = True) -> None:
        """
        A custom trainer that wraps the pytorch lightning's trainer used for, but not only, multilingual training.
        :param tokenizer: the tokenizer that will be used by the translation datasets.
        :param train_steps: the number of training updates that the trainer will perform (default=100000).
        :param val_every_n_steps: how often to check the validation set (default=10000).
        :param log_every_n_steps: how often to log the metrics (default=500).
        :param dataloader_num_workers: the number of workers used by both the train and validation dataloaders
            (default=8).
        :param log_directory: the log directory in which to save the logs, "" corresponds to the current working
            directory (default="").
        :param batch_type: the type of batch sampler used by the train dataloader, can be either "heterogeneous" (each
            batch will have one single translation direction) or "homogeneous" (each batch will contain more than
            one translation direction) (default="heterogeneous").
        :param use_wandb: whether to use wandb as a logger, otherwise tensorboard will be used (default=True).
        """
        super().__init__(tokenizer, train_steps, val_every_n_steps, log_every_n_steps, dataloader_num_workers,
                         log_directory, batch_type, use_wandb)

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
        if self.batch_type == "heterogeneous":
            batch_sampler = HeterogeneousSampler(train_dataset, train_bsz, True)
        else:
            batch_sampler = HomogeneousSampler(train_dataset, train_bsz, True)

        return super()._build_dataloaders_base_(model, train_dataset, batch_sampler, val_datasets, val_bsz)

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
            logger = WandbLogger(logger_version, self.log_directory + "/logs", project="ContinualNAT",
                                 version=logger_version)
        else:
            logger = TensorBoardLogger(self.log_directory, name="logs", version=logger_version)

        checkpoint = ModelCheckpoint(save_top_k=2, monitor="mean_BLEU", mode="max", save_on_train_epoch_end=False)
        prog_bar_theme = RichProgressBarTheme(description="red", progress_bar="dark_blue",
                                              progress_bar_finished="dark_blue", progress_bar_pulse="dark_blue")
        prog_bar = RichProgressBar(theme=prog_bar_theme)

        # noinspection DuplicatedCode
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer_callbacks = [checkpoint, lr_monitor, prog_bar]

        # Train the model
        trainer = Trainer(devices=1, precision="16-mixed", logger=logger, max_steps=self.train_steps,
                          val_check_interval=self.val_every_n_steps * accumulation_steps,
                          check_val_every_n_epoch=None, log_every_n_steps=self.log_every_n_steps,
                          accumulate_grad_batches=accumulation_steps, gradient_clip_val=1.0,
                          callbacks=trainer_callbacks)
        trainer.fit(model, train_dataloader, val_dataloaders)


class ContinualTrainer(TrainerCore):

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 buffer_size: int,
                 keep_previous_examples: bool = True,
                 train_steps: int = 100000,
                 val_every_n_steps: int = 10000,
                 log_every_n_steps: int = 500,
                 dataloader_num_workers: int = 8,
                 log_directory: str = "",
                 exp_batch_type: str = "heterogeneous",
                 use_wandb: bool = True) -> None:
        """
        A custom trainer that wraps the pytorch lightning's trainer.
        :param tokenizer: the tokenizer that will be used by the translation datasets.
        :param train_steps: the number of training updates that the trainer will perform (default=100000).
        :param val_every_n_steps: how often to check the validation set (default=10000).
        :param log_every_n_steps: how often to log the metrics (default=500).
        :param dataloader_num_workers: the number of workers used by both the train and validation dataloaders
            (default=8).
        :param log_directory: the log directory in which to save the logs, "" corresponds to the current working
            directory (default="").
        :param exp_batch_type: the type of batch sampler used by the experiences, can be either "heterogeneous" (each
            batch will have one single translation direction) or "homogeneous" (each batch will contain more than
            one translation direction) (default="heterogeneous").
        :param use_wandb: whether to use wandb as a logger, otherwise tensorboard will be used (default=True).
        """
        super().__init__(tokenizer, train_steps, val_every_n_steps, log_every_n_steps, dataloader_num_workers,
                         log_directory, exp_batch_type, use_wandb)
        self.exp_batch_type = self.batch_type
        del self.batch_type
        self.buffer = Buffer(buffer_size, keep_previous_examples)

    def __build_dataloaders(self,
                            model: TransformerCore,
                            exp_idx: int,
                            exp_datasets: List[TranslationDataset],
                            val_datasets: List[TranslationDataset],
                            train_bsz: int = 128,
                            val_bsz: int = 128) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        train_dataset = ConcatDataset(exp_datasets)
        if self.exp_batch_type == "heterogeneous":
            exp_sampler = HeterogeneousSampler(train_dataset, train_bsz // 2, True)
        else:
            exp_sampler = HomogeneousSampler(train_dataset, train_bsz // 2, True)

        if exp_idx != 0:
            mem_sampler = BatchSampler(RandomSampler(self.buffer), train_bsz // 2, True)
            batch_sampler = ExperienceReplaySampler(exp_sampler, mem_sampler)
        else:
            batch_sampler = exp_sampler

        return super()._build_dataloaders_base_(model, train_dataset, batch_sampler, val_datasets, val_bsz)

    def __compute_logger_version(self, model, exp_idx) -> str:
        logger_version = f"{model.__class__.__name__}"
        v_num = 0
        while os.path.exists(f"logs/{logger_version}_{v_num}"):
            v_num += 1

        logger_version += f"_{v_num}_exp{exp_idx}"
        return logger_version

    def train(self,
              model: TransformerCore,
              exps: List[Union[TranslationDataset, List[TranslationDataset]]],
              val_datasets: list[TranslationDataset],
              train_bsz: int = 128,
              val_bsz: int = 128,
              tokens_per_batch: int | None = None,
              logger_version: str | None = None,
              save_model_each_exp: bool = False) -> None:
        for i, exp in enumerate(exps):
            current_exp = [exp] if isinstance(exp, TranslationDataset) else exp

            # Dataloaders
            train_dataloader, val_dataloaders = self.__build_dataloaders(model, i, current_exp, val_datasets, train_bsz,
                                                                         val_bsz)

            # Accumulation steps
            max_length = current_exp[0].tokenizer_state["max_length"]
            tokens_per_batch = tokens_per_batch if tokens_per_batch is not None else train_bsz * max_length
            accumulation_steps = compute_accumulation_steps(train_bsz, max_length, tokens_per_batch)

            # Logger and other train callbacks
            if logger_version is None:
                logger_version = self.__compute_logger_version(model, i)
            else:
                logger_version = f"{logger_version}_exp{i}"

            checkpoint = ModelCheckpoint(save_top_k=2, monitor="mean_BLEU", mode="max", save_on_train_epoch_end=False)
            prog_bar_theme = RichProgressBarTheme(description="red", progress_bar="dark_blue",
                                                  progress_bar_finished="dark_blue", progress_bar_pulse="dark_blue")
            prog_bar = RichProgressBar(theme=prog_bar_theme)
            if self.use_wandb:
                logger = WandbLogger(logger_version, self.log_directory + "/logs", project="ContinualNAT",
                                     version=logger_version)
            else:
                logger = TensorBoardLogger(self.log_directory, name="logs", version=logger_version)

            # noinspection DuplicatedCode
            lr_monitor = LearningRateMonitor(logging_interval='step')
            trainer_callbacks = [checkpoint, lr_monitor, prog_bar]

            # Train the model
            trainer = Trainer(devices=1, precision="16-mixed", logger=logger, max_steps=self.train_steps,
                              val_check_interval=self.val_every_n_steps * accumulation_steps,
                              check_val_every_n_epoch=None, log_every_n_steps=self.log_every_n_steps,
                              accumulate_grad_batches=accumulation_steps, gradient_clip_val=1.0,
                              callbacks=trainer_callbacks)
            trainer.fit(model, train_dataloader, val_dataloaders)

            # Add experience to the buffer
            self.buffer.add_experience(exp)

            if save_model_each_exp:
                torch.save(model.state_dict(), f"/disk1/a.ristori/models/{logger_version}")