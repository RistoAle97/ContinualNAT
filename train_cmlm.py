import torch
import yaml
from transformers import MBartTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme
from src.data import *
from src.models import *
from src.utils import SUPPORTED_LANGUAGES, compute_accumulation_steps
from typing import Dict, Union


if __name__ == "__main__":
    # Set-up
    torch.set_float32_matmul_precision("medium")

    # Retrieve configurations
    with open("train.yaml") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    tokenizer_parameters: Dict[str, Union[str, int]] = config["tokenizer"]
    dataset_parameters: Dict[str, str] = config["dataset"]
    model_parameters: Dict[str, Union[str, int]] = config["model"]
    src_lang: str = config["src_lang"]
    tgt_lang: str = config["tgt_lang"]
    train_batch_size: int = config["train_batch_size"]
    val_batch_size: int = config["val_batch_size"]
    max_length: int = config["max_length"]
    padding: str = config["padding"]
    tokens_per_batch: int = config["tokens_per_batch"]
    accumulate_gradient: bool = config["accumulate_gradient"]
    log_step: int = config["log_step"]
    training_updates: int = config["training_updates"]

    # Tokenizer
    src_tokenizer = SUPPORTED_LANGUAGES[src_lang]
    tgt_tokenizer = SUPPORTED_LANGUAGES[tgt_lang]
    tokenizer = MBartTokenizerFast(**tokenizer_parameters, src_lang=src_tokenizer, tgt_lang=tgt_tokenizer)
    print(f"Retrieved {tokenizer.__class__.__name__} with vocab size: {len(tokenizer)}")

    # Dataset
    dataset = load_dataset(**dataset_parameters)
    if "validation" not in dataset.keys():
        dataset_split = dataset["train"].train_test_split(test_size=3000)
        dataset_train = dataset_split["train"]
        dataset_val = dataset_split["test"]
    else:
        dataset_train = dataset["train"]
        dataset_val = dataset["validation"]

    dataset_train = TranslationDataset(src_lang, tgt_lang, dataset_train)
    collator_train = BatchCollatorCMLM(tokenizer, max_length=max_length, padding=padding, train=True)
    dataloader_train = DataLoader(dataset_train, train_batch_size, num_workers=8, collate_fn=collator_train,
                                  pin_memory=True, drop_last=True)
    dataset_val = TranslationDataset(src_lang, tgt_lang, dataset_val)
    collator_val = BatchCollatorCMLM(tokenizer, max_length=max_length, padding=padding, train=False)
    dataloader_val = DataLoader(dataset_val, val_batch_size, num_workers=8, collate_fn=collator_val, pin_memory=True)

    # Model
    cmlm_config = CMLMConfig(len(tokenizer), **model_parameters, sos_token_id=tokenizer.bos_token_id,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                             mask_token_id=tokenizer.mask_token_id, length_token_id=tokenizer.cls_token_id)
    cmlm_model = CMLM(cmlm_config)

    # Compute accumulation steps if requested by the user
    accumulation_steps = 1
    if accumulate_gradient:
        accumulation_steps = compute_accumulation_steps(train_batch_size, max_length, tokens_per_batch)

    # Logger and model checkpoints
    logger = TensorBoardLogger("", name="logs", version=f"{cmlm_model.__class__.__name__}_{src_lang}_{tgt_lang}_1")
    progr_bar_theme = RichProgressBarTheme(description="red", progress_bar="dark_blue",
                                           progress_bar_finished="dark_blue", progress_bar_pulse="dark_blue")
    progr_bar = RichProgressBar(theme=progr_bar_theme)
    checkpoint_call = ModelCheckpoint(save_top_k=2, monitor="train_loss", every_n_train_steps=10000)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Train the model
    trainer = Trainer(devices=1, precision="16-mixed", logger=logger, max_steps=training_updates,
                      log_every_n_steps=log_step, accumulate_grad_batches=accumulation_steps, gradient_clip_val=1.0,
                      callbacks=[checkpoint_call, lr_monitor, progr_bar])
    trainer.fit(cmlm_model, dataloader_train, dataloader_val)