import torch
import math
import yaml
from transformers import MBartTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from src.data import *
from src.models import *
from typing import Dict


if __name__ == "__main__":
    # Set-up
    torch.set_float32_matmul_precision("medium")

    # Retrieve configurations
    with open("train.yaml") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    dataset_parameters: Dict[str, str] = config["dataset"]
    src_lang: str = config["src_lang"]
    tgt_lang: str = config["tgt_lang"]
    batch_size: int = config["batch_size"]
    max_length: int = config["max_length"]
    padding: str = config["padding"]
    warmup_steps: int = config["warmup_steps"]
    tokens_per_batch: int = config["tokens_per_batch"]
    accumulate_gradient: bool = config["accumulate_gradient"]
    log_update_step: int = config["log_step"]
    training_updates: int = config["training_updates"]

    # Tokenizer
    tokenizer = MBartTokenizerFast(tokenizer_file="tokenizers/sp_32k.json", cls_token="<length>",
                                   src_lang="en_XX", tgt_lang="de_DE")
    print(f"Retrieved {tokenizer.__class__.__name__} with vocab size: {len(tokenizer)}")

    # Dataset
    dataset = load_dataset(**dataset_parameters, verification_mode="no_checks")
    if "validation" not in dataset.keys():
        dataset_split = dataset["train"].train_test_split(test_size=3000)
        dataset_train = dataset_split["train"]
        dataset_val = dataset_split["test"]
    else:
        dataset_train = dataset["train"]
        dataset_val = dataset["validation"]

    dataset_train = TranslationDataset(src_lang, tgt_lang, dataset_train)
    batch_collator_train = BatchCollator(tokenizer, max_length=max_length, padding=padding)
    dataloader_train = DataLoader(dataset_train, batch_size, num_workers=8, collate_fn=batch_collator_train,
                                  pin_memory=True, drop_last=True)
    dataset_val = TranslationDataset(src_lang, tgt_lang, dataset_val)
    batch_collator_val = BatchCollator(tokenizer, max_length=max_length, padding=padding)
    dataloader_validation = DataLoader(dataset_val, 32, num_workers=8, collate_fn=batch_collator_val, pin_memory=True)

    # Model
    transformer = Transformer(len(tokenizer))

    # Compute accumulation steps if requested by the user
    actual_tokens_per_batch = batch_size * max_length
    if accumulate_gradient and tokens_per_batch > actual_tokens_per_batch:
        accumulation_steps = math.ceil(tokens_per_batch / actual_tokens_per_batch)
    else:
        accumulation_steps = 1

    # Logger and model checkpoints
    logger = TensorBoardLogger("", name="logs")
    checkpoint_call = ModelCheckpoint(save_top_k=2, monitor="train_loss", every_n_train_steps=10000)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Train the model
    trainer = Trainer(devices=1, precision="16-mixed", logger=logger, max_steps=100000, val_check_interval=10000,
                      log_every_n_steps=500, accumulate_grad_batches=accumulation_steps, gradient_clip_val=1.0,
                      callbacks=[checkpoint_call, lr_monitor])
    trainer.fit(transformer, dataloader_train, dataloader_validation)
