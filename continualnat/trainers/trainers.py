import os
from typing import Dict, List, Set, Tuple, Union

import torch
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar, EarlyStopping
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
from continualnat.models.glat.glat import GLAT
from continualnat.utils.utils import MBART_LANG_MAP, compute_accumulation_steps


class TrainerCore:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        train_steps: int = 100000,
        val_every_n_steps: int = 10000,
        log_every_n_steps: int = 500,
        dataloader_num_workers: int = 8,
        log_directory: str = "",
        batch_type: str = None,
        use_wandb: bool = True,
    ) -> None:
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
        :param batch_type: the type of batch sampler used by the train dataloader. It can be None (the training datasets
            will be simply concatenated into one and shuffled), "heterogeneous" (each batch will have one single
            translation direction) or "homogeneous" (each batch will contain more than one translation direction). The
            None value is useful in a scenario where all the datasets have the same size (default=None).
        :param use_wandb: whether to use wandb as a logger, otherwise tensorboard will be used (default=True).
        """
        self.tokenizer = tokenizer
        self.train_steps = train_steps
        self.val_every_n_steps = val_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.num_workers = dataloader_num_workers
        self.log_directory = log_directory
        if batch_type not in [None, "heterogeneous", "homogeneous"]:
            raise ValueError("The batch type should be \"heterogeneous\" or \"homogeneous\"")

        self.batch_type = batch_type
        self.use_wandb = use_wandb

    def _build_dataloaders_base(
        self,
        model: TransformerCore,
        train_dataset: ConcatDataset[TranslationDataset],
        train_batch_sampler: Union[BatchSamplerCore, ExperienceReplaySampler],
        val_datasets: List[TranslationDataset],
        train_bsz: int = 128,
        val_bsz: int = 128,
    ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        # Define the batch collator parameters
        batch_collator_args = {
            "is_mlm": False,
            "shift_lang_token": True,
            "return_lengths": False,
            "pad_token_id": model.pad_token_id,
            "mask_token_id": model.mask_token_id if hasattr(model, "mask_token_id") else self.tokenizer.mask_token_id,
            "p_masking": 0.0,
        }
        if isinstance(model, (CMLM, GLAT)):
            batch_collator_args["is_mlm"] = True
            batch_collator_args["shift_lang_token"] = False
            batch_collator_args["return_lengths"] = True
            if isinstance(model, CMLM):
                batch_collator_args["p_masking"] = "random" if not model.glat_training else 1.0
            else:
                batch_collator_args["p_masking"] = 0.0

        # Train batch collator and dataloader
        batch_collator_train = BatchCollator(**batch_collator_args)
        dataloader_args = {"num_workers": self.num_workers, "collate_fn": batch_collator_train, "pin_memory": True}
        if self.batch_type is not None:
            # Use one of the heterogeneous or homogeneous sampler
            train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **dataloader_args)
        else:
            # Shuffle the concatenated datasets
            train_dataloader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True, **dataloader_args)

        # Validation dataloaders (one for each translation direction)
        val_dataloaders = {}
        batch_collator_args["p_masking"] = 1.0 if isinstance(model, CMLM) else 0.0
        for dataset in val_datasets:
            src_lang = dataset.src_lang
            tgt_lang = dataset.tgt_lang
            batch_collator_val = BatchCollator(**batch_collator_args)
            dataloader_args["collate_fn"] = batch_collator_val
            val_dataloaders[f"{src_lang}_{tgt_lang}"] = DataLoader(dataset, batch_size=val_bsz, **dataloader_args)

        return train_dataloader, val_dataloaders

    def __build_dataloaders(self, *kwargs) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        raise NotImplementedError

    def __compute_logger_version(self, *kwargs) -> str:
        raise NotImplementedError

    def train(self, *kwargs) -> None:
        raise NotImplementedError


class MultilingualTrainer(TrainerCore):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        train_steps: int = 100000,
        val_every_n_steps: int = 10000,
        log_every_n_steps: int = 500,
        dataloader_num_workers: int = 8,
        log_directory: str = "",
        batch_type: str = None,
        use_wandb: bool = True,
    ) -> None:
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
        :param batch_type: the type of batch sampler used by the train dataloader. It can be None (the training datasets
            will be simply concatenated into one and shuffled), "heterogeneous" (each batch will have one single
            translation direction) or "homogeneous" (each batch will contain more than one translation direction). The
            None value is useful in a scenario where all the datasets have the same size (default=None).
        :param use_wandb: whether to use wandb as a logger, otherwise tensorboard will be used (default=True).
        """
        super().__init__(
            tokenizer=tokenizer,
            train_steps=train_steps,
            val_every_n_steps=val_every_n_steps,
            log_every_n_steps=log_every_n_steps,
            dataloader_num_workers=dataloader_num_workers,
            log_directory=log_directory,
            batch_type=batch_type,
            use_wandb=use_wandb,
        )

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

    def __build_dataloaders(
        self,
        model: TransformerCore,
        train_datasets: List[TranslationDataset],
        val_datasets: List[TranslationDataset],
        train_bsz: int = 128,
        val_bsz: int = 128,
    ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        # Train dataloader
        train_dataset = ConcatDataset(train_datasets)
        if self.batch_type == "heterogeneous":
            batch_sampler = HeterogeneousSampler(train_dataset, train_bsz, True)
        elif self.batch_type == "homogeneous":
            batch_sampler = HomogeneousSampler(train_dataset, train_bsz, True)
        else:
            batch_sampler = None

        return super()._build_dataloaders_base(model, train_dataset, batch_sampler, val_datasets, train_bsz, val_bsz)

    @staticmethod
    def __compute_logger_version(
        model: TransformerCore,
        nmt_directions: Set[str],
        lang_pairs: Set[Tuple[str, str]],
    ) -> str:
        # First, take the model class as start point
        logger_version = f"{model.__class__.__name__}"

        # Define how many language pairs are involved
        if len(lang_pairs) > 1:
            logger_version += "_multilingual"
        else:
            first_lang, second_lang = list(nmt_directions)[0].split("-")
            logger_version += f"_{first_lang}_{second_lang}"
            if len(nmt_directions) > 1:
                logger_version += "_both"

        # Add a progressive number in order to avoid duplicated versions
        v_num = 0
        while os.path.exists(f"logs/{logger_version}_{v_num}"):
            v_num += 1

        logger_version += f"_{v_num}"

        return logger_version

    def train(
        self,
        model: TransformerCore,
        train_datasets: List[TranslationDataset],
        val_datasets: List[TranslationDataset],
        train_bsz: int = 128,
        val_bsz: int = 128,
        tokens_per_batch: int = None,
        logger_version: str = None,
        early_stopping: bool = False,
        patience: int = 5,
        wandb_project: str = None,
    ) -> None:
        """
        Trains and validates the model given a list of training and validation datasets.
        :param model: the model to train.
        :param train_datasets: a list containing all the datasets used during training.
        :param val_datasets: a list containing all the datasets used during validation.
        :param train_bsz: the train batch size (default=128).
        :param val_bsz: the validation batch size (default=128).
        :param tokens_per_batch: the approximated number of tokens that each batch should contain. If None, then no
            accmulation steps will be performed (default=None).
        :param logger_version: the version used by the logger. If None, then its value will be modelclass_type_seq
            (default=None).
        :param early_stopping: whether to use early stopping if the mean BLEU does not improve for a fixed number
            of validation epochs (default=False).
        :param patience: the patience parameter used by the early stopping callback (default=5).
        :param wandb_project: the wandb project to which the run will belong to. If None, then "lightning_logs"
            will be used (default=None).
        """
        # Build translation directions and unique lang pairs
        nmt_directions, lang_pairs = self.__build_nmt_directions(train_datasets)

        # Dataloaders
        train_dataloader, val_dataloaders = self.__build_dataloaders(
            model, train_datasets, val_datasets, train_bsz, val_bsz
        )

        # Accumulation steps
        max_length = train_datasets[0].max_length
        tokens_per_batch = tokens_per_batch if tokens_per_batch is not None else train_bsz * max_length
        accumulation_steps = compute_accumulation_steps(train_bsz, max_length, tokens_per_batch)

        # Logger
        if logger_version is None:
            logger_version = self.__compute_logger_version(model, nmt_directions, lang_pairs)

        if self.use_wandb:
            wandb_project = "wandb_logs" if wandb_project is None else wandb_project
            logger = WandbLogger(
                name=logger_version,
                save_dir=self.log_directory + "/logs",
                project=wandb_project,
                version=logger_version,
            )
        else:
            logger = TensorBoardLogger(self.log_directory, name="logs", version=logger_version)

        # Model checkpointing
        checkpoint = ModelCheckpoint(save_top_k=2, monitor="mean_BLEU", mode="max", save_on_train_epoch_end=False)

        # Progress bar
        prog_bar_theme = RichProgressBarTheme(
            description="red",
            progress_bar="dark_blue",
            progress_bar_finished="dark_blue",
            progress_bar_pulse="dark_blue",
        )
        prog_bar = RichProgressBar(theme=prog_bar_theme)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer_callbacks = [checkpoint, lr_monitor, prog_bar]

        # Early stopping
        if early_stopping:
            early_stopping_callback = EarlyStopping("mean_BLEU", patience=patience, mode="max")
            trainer_callbacks.append(early_stopping_callback)

        # Train the model
        trainer = Trainer(
            devices=1,
            precision="16-mixed",
            logger=logger,
            max_steps=self.train_steps,
            val_check_interval=self.val_every_n_steps * accumulation_steps,
            check_val_every_n_epoch=None,
            log_every_n_steps=self.log_every_n_steps,
            accumulate_grad_batches=accumulation_steps,
            gradient_clip_val=1.0,
            callbacks=trainer_callbacks,
        )
        trainer.fit(model, train_dataloader, val_dataloaders)


class ContinualTrainer(TrainerCore):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        buffer_size: int,
        keep_previous_examples: bool = True,
        train_steps: int = 100000,
        val_every_n_steps: int = 10000,
        log_every_n_steps: int = 500,
        dataloader_num_workers: int = 8,
        log_directory: str = "",
        batch_type: str = None,
        buffer_batch_size: float = None,
        use_wandb: bool = True,
    ) -> None:
        """
        A custom trainer that wraps the pytorch lightning's trainer.
        :param tokenizer: the tokenizer that will be used by the translation datasets.
        :param buffer_size: the buffer size, if 0 the training will be incremental.
        :param keep_previous_examples: whether to keep the previos examples of the buffer for future experiences
            (default=True).
        :param train_steps: the number of training updates that the trainer will perform (default=100000).
        :param val_every_n_steps: how often to check the validation set (default=10000).
        :param log_every_n_steps: how often to log the metrics (default=500).
        :param dataloader_num_workers: the number of workers used by both the train and validation dataloaders
            (default=8).
        :param log_directory: the log directory in which to save the logs, "" corresponds to the current working
            directory (default="").
        :param batch_type: the type of batch sampler used by the experiences. It can be None (the training datasets
            will be simply concatenated into one and shuffled), "heterogeneous" (each batch will have one single
            translation direction) or "homogeneous" (each batch will contain more than one translation direction). The
            None value is useful in a scenario where all the datasets have the same size (default=None).
        :param buffer_batch_size: the percentage of examples inside a batch that should come from the buffer. If None
            then no sampler will be used and the buffer will be simply concatenated to the current experience
            (default=None).
        :param use_wandb: whether to use wandb as a logger, otherwise tensorboard will be used (default=True).
        """
        super().__init__(
            tokenizer=tokenizer,
            train_steps=train_steps,
            val_every_n_steps=val_every_n_steps,
            log_every_n_steps=log_every_n_steps,
            dataloader_num_workers=dataloader_num_workers,
            log_directory=log_directory,
            batch_type=batch_type,
            use_wandb=use_wandb,
        )
        # Buffer
        self.buffer = Buffer(buffer_size, keep_previous_examples) if buffer_size > 0 else None
        if buffer_batch_size is not None and not 0.0 <= buffer_batch_size <= 1.0:
            raise ValueError("The percentage of examples coming from the buffer should be in [0.0, 1.0].")

        self.buffer_batch_size = buffer_batch_size

    def __build_dataloaders(
        self,
        model: TransformerCore,
        exp_idx: int,
        exp_datasets: List[TranslationDataset],
        val_datasets: List[TranslationDataset],
        train_bsz: int = 128,
        val_bsz: int = 128,
    ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        if self.buffer is not None and self.buffer_batch_size is not None:
            train_dataset = ConcatDataset(exp_datasets)
            exp_batch_examples = train_bsz * (1.0 - self.buffer_batch_size)
            mem_batch_examples = train_bsz - exp_batch_examples

            # Experience batch sampler
            if self.batch_type == "heterogeneous":
                exp_sampler = HeterogeneousSampler(train_dataset, exp_batch_examples, True)
            elif self.batch_type == "homogeneous":
                exp_sampler = HomogeneousSampler(train_dataset, exp_batch_examples, True)
            else:
                exp_sampler = RandomSampler(train_dataset)

            # Memory batch sampler
            if exp_idx != 0 and self.buffer_batch_size is not None:
                mem_sampler = BatchSampler(RandomSampler(self.buffer), mem_batch_examples, True)
                batch_sampler = ExperienceReplaySampler(exp_sampler, mem_sampler)
            else:
                batch_sampler = exp_sampler

        else:
            # If the percentage of examples coming from the buffer is None, then we concatenate the buffer's datasets
            # with the current experience and no batch sampler will be employed, doesn't matter which batch type
            # is used.
            batch_sampler = None
            if self.buffer is not None:
                exp_datasets.extend(self.buffer.list_datasets())

            train_dataset = ConcatDataset(exp_datasets)

        return super()._build_dataloaders_base(model, train_dataset, batch_sampler, val_datasets, train_bsz, val_bsz)

    def __compute_logger_version(self, model, exp_idx) -> str:
        logger_version = f"{model.__class__.__name__}"
        v_num = 0
        while os.path.exists(f"logs/{logger_version}_{v_num}"):
            v_num += 1

        logger_version += f"_{v_num}_exp{exp_idx}"
        return logger_version

    def train(
        self,
        model: TransformerCore,
        exps: List[Union[TranslationDataset, List[TranslationDataset]]],
        val_datasets: List[Union[TranslationDataset, List[TranslationDataset]]],
        train_bsz: int = 128,
        val_bsz: int = 128,
        tokens_per_batch: int | None = None,
        logger_version: str | None = None,
        save_model_each_exp: bool = False,
        wandb_project: str = None,
    ) -> None:
        """
        Trains the models given a list of training datasets while validating its performances.
        :param model: the model to train.
        :param exps: a list containing all the experiences used during training.
        :param val_datasets: a list containing all the datasets used during validation divided by experience.
        :param train_bsz: the train batch size (default=128).
        :param val_bsz: the validation batch size (default=128).
        :param tokens_per_batch: the approximated number of tokens that each batch should contain. If None, then no
            accmulation steps will be performed (default=None).
        :param logger_version: the version used by the logger. If None, then its value will be modelclass_seq
            (default=None).
        :param save_model_each_exp: whether to save the model at the end of each experience (default=False).
        :param wandb_project: the wandb project to which the run will belong to. If None, then "lightning_logs"
            will be used (default=None).
        """
        for i, exp in enumerate(exps):
            current_exp = [exp] if isinstance(exp, TranslationDataset) else exp
            val_exp = val_datasets[i]
            val_exp = [val_exp] if isinstance(val_exp, TranslationDataset) else val_exp

            # Dataloaders
            train_dataloader, val_dataloaders = self.__build_dataloaders(
                model, i, current_exp, val_exp, train_bsz, val_bsz
            )

            # Accumulation steps
            max_length = current_exp[0].max_length
            tokens_per_batch = tokens_per_batch if tokens_per_batch is not None else train_bsz * max_length
            accumulation_steps = compute_accumulation_steps(train_bsz, max_length, tokens_per_batch)

            # Logger and other train callbacks
            if logger_version is None:
                logger_version_train = self.__compute_logger_version(model, i)
            else:
                logger_version_train = f"{logger_version}_exp{i}"

            checkpoint = ModelCheckpoint(save_top_k=2, monitor="mean_BLEU", mode="max", save_on_train_epoch_end=False)
            prog_bar_theme = RichProgressBarTheme(
                description="red",
                progress_bar="dark_blue",
                progress_bar_finished="dark_blue",
                progress_bar_pulse="dark_blue",
            )
            prog_bar = RichProgressBar(theme=prog_bar_theme)
            if self.use_wandb:
                wandb_project = "wandb_logs" if wandb_project is None else wandb_project
                logger = WandbLogger(
                    name=logger_version_train,
                    save_dir=self.log_directory + "/logs",
                    project=wandb_project,
                    version=logger_version_train,
                )
            else:
                logger = TensorBoardLogger(self.log_directory, name="logs", version=logger_version_train)

            lr_monitor = LearningRateMonitor(logging_interval="step")
            trainer_callbacks = [checkpoint, lr_monitor, prog_bar]

            # Train the model
            trainer = Trainer(
                devices=1,
                precision="16-mixed",
                logger=logger,
                max_steps=self.train_steps,
                val_check_interval=self.val_every_n_steps * accumulation_steps,
                check_val_every_n_epoch=None,
                log_every_n_steps=self.log_every_n_steps,
                accumulate_grad_batches=accumulation_steps,
                gradient_clip_val=1.0,
                callbacks=trainer_callbacks,
            )
            trainer.fit(model, train_dataloader, val_dataloaders)

            # Add experience to the buffer if it is not the last one
            if self.buffer is not None and i != len(exps) - 1:
                self.buffer.add_experience(exp)

            # Close the wandb logger
            if isinstance(logger, WandbLogger):
                wandb.finish()

            # Save the model after each experience if the user wishes so
            if save_model_each_exp:
                torch.save(model.state_dict(), f"/disk1/a.ristori/models/{logger_version_train}")
