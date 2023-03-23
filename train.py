import torch
import math
import yaml
from transformers import MBartTokenizer
from datasets import load_dataset
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from src.data import *
from src.models import *
from src.utils import model_size, model_n_parameters, generate_causal_mask, compute_lr, SUPPORTED_LANGUAGES
from typing import Dict, Union


if __name__ == "__main__":
    # Set-up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

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
    accumulation_steps: Union[int, str] = config["accumulation_steps"]
    log_update_step: int = config["log_step"]
    validation_step: bool = config["validation_step"]
    training_updates: int = config["training_updates"]

    # Tokenizer
    tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained(config["tokenizer"],
                                                               src_lang=SUPPORTED_LANGUAGES[src_lang],
                                                               tgt_lang=SUPPORTED_LANGUAGES[tgt_lang],
                                                               cache_dir="/disk1/a.ristori/tokenizers")
    print(f"Retrieved {tokenizer.__class__.__name__} with vocab size: {len(tokenizer)}\n")

    # Dataset
    dataset_train = load_dataset(**dataset_parameters, split="train", verification_mode="no_checks")
    dataset_train = TranslationDataset(src_lang, tgt_lang, dataset_train)
    batch_collator_train = BatchCollator(tokenizer, max_length=max_length, padding=padding)
    dataloader_train = DataLoader(dataset_train, batch_size, collate_fn=batch_collator_train, drop_last=True)
    # if validation_step:
    dataset_validation = load_dataset(**dataset_parameters, split="validation", verification_mode="no_checks")
    dataset_validation = TranslationDataset(src_lang, tgt_lang, dataset_validation)
    batch_collator_validation = BatchCollator(tokenizer, max_length=max_length, padding=padding)
    dataloader_validation = DataLoader(dataset_validation, batch_size, collate_fn=batch_collator_validation)

    # Model
    transformer = Transformer(len(tokenizer)).to(device)
    n_parameters, n_trainable_parameters = model_n_parameters(transformer)
    transformer_size = model_size(transformer)
    print(f"\nUsing {transformer.__class__.__name__} model:")
    print(f"\tParameters: {n_parameters}\n"
          f"\tTrainable parameters: {n_trainable_parameters}\n"
          f"\tSize: {transformer_size}\n")

    # Define loss function, optimizer and learning rate scheduler
    pad_token = tokenizer.pad_token_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=config["label_smoothing"])
    optimizer = AdamW(transformer.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lambda steps: compute_lr(steps, transformer.d_model, config["warmup_steps"]))

    # Set accumulation steps to match the required number of tokens per batch if none was passed to the former
    if accumulation_steps == "None":
        actual_tokens_per_batch = batch_size * max_length
        if tokens_per_batch > actual_tokens_per_batch:
            accumulation_steps = math.ceil(tokens_per_batch / actual_tokens_per_batch)
        else:
            accumulation_steps = 1

    with tqdm(desc=f"{transformer.__class__.__name__} training", total=training_updates) as pbar:
        accumulation_loss = 0.0
        current_step = 0
        n_updates = 0
        board = SummaryWriter()

        # Train loop
        while n_updates < training_updates:
            for train_batch in dataloader_train:
                transformer.train()
                # Retrieve encoder inputs, labels and decoder inputs
                input_ids = train_batch["input_ids"].to(device)
                labels = train_batch["labels"].to(device)
                decoder_input_ids = train_batch["decoder_input_ids"].to(device)

                # Create masks
                e_pad_mask = (input_ids == pad_token).to(device)
                d_pad_mask = (decoder_input_ids == pad_token).to(device)
                d_mask = generate_causal_mask(decoder_input_ids.shape[-1]).to(device)

                # Compute logits and loss
                with torch.cuda.amp.autocast():
                    logits = transformer(input_ids, decoder_input_ids, d_mask, e_pad_mask, d_pad_mask)
                    loss = loss_fn(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))

                # Backward pass
                accumulation_loss += loss.item()
                loss /= accumulation_steps
                loss.backward()

                # Update weights and do one step for both the optimizer and scheduler every accumulation_steps
                if (current_step + 1) % accumulation_steps == 0:
                    clip_grad_norm_(transformer.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Udpate the progress bar and reset accumulation loss
                    pbar.update(1)
                    mean_loss = accumulation_loss / accumulation_steps
                    pbar.set_postfix(loss=str(mean_loss)[0:6])
                    n_updates += 1
                    accumulation_loss = 0

                    # Log loss and learning rate on the tensorboard
                    if n_updates % log_update_step == 0:
                        board.add_scalar("Loss/train", mean_loss, n_updates)
                        board.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], n_updates)

                        # Validation step
                        if validation_step:
                            transformer.eval()
                            sum_validation_loss = 0.0
                            for validation_batch in dataloader_validation:
                                input_ids_validation = validation_batch["input_ids"].to(device)
                                labels_validation = validation_batch["labels"].to(device)
                                decoder_input_ids_validation = validation_batch["decoder_input_ids"].to(device)

                                # Validation masks
                                e_pad_mask_validation = (input_ids_validation == pad_token).to(device)
                                d_pad_mask_validation = (decoder_input_ids_validation == pad_token).to(device)
                                d_mask_validation = generate_causal_mask(decoder_input_ids.shape[-1]).to(device)

                                with torch.cuda.amp.autocast():
                                    validation_logits = transformer(input_ids_validation, decoder_input_ids_validation,
                                                                    d_mask_validation, e_pad_mask_validation,
                                                                    d_pad_mask_validation)
                                    validation_loss = loss_fn(validation_logits.contiguous()
                                                              .view(-1, validation_logits.size(-1)),
                                                              labels_validation.contiguous().view(-1))

                                sum_validation_loss += validation_loss.item()

                            board.add_scalar("Loss/validation", sum_validation_loss / len(dataset_validation),
                                             n_updates)

                    # Check stopping criteria
                    if n_updates == training_updates:
                        break

                current_step += 1

    # Close tensorboard and save the model
    board.flush()
    board.close()
    torch.save(transformer, f"/home/a.ristori/ContinualNAT/{transformer.__class__.__name__}_{src_lang}_{tgt_lang}")
