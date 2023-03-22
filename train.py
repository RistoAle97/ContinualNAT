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


if __name__ == "__main__":
    # Set-up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Retrieve configurations
    with open("config.yaml") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    huggingface_dataset = config["dataset"]
    src_lang = config["src_lang"]
    tgt_lang = config["tgt_lang"]
    batch_size = config["batch_size"]
    max_length = config["max_length"]
    padding = config["padding"]
    warmup_steps = config["warmup_steps"]
    tokens_per_batch = config["tokens_per_batch"]
    accumulation_steps = config["accumulation_steps"]
    training_updates = config["training_udpates"]

    # Tokenizer
    tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained("tokenizers/mbart_tokenizer_cmlm",
                                                               src_lang=SUPPORTED_LANGUAGES[src_lang],
                                                               tgt_lang=SUPPORTED_LANGUAGES[tgt_lang])
    print(f"Retrieved {tokenizer.__class__.__name__} with vocab size: {len(tokenizer)}\n")

    # Dataset
    dataset = load_dataset("yhavinga/ccmatrix", f"{src_lang}-{tgt_lang}",
                           cache_dir=f"D:/MasterDegreeThesis/datasets/ccmatrix_{src_lang}_{tgt_lang}",
                           split="train[:4096]", verification_mode="no_checks")
    # dataset = dataset.train_test_split(test_size=196)
    dataset_train = TranslationDataset(src_lang, tgt_lang, dataset)
    batch_collator_train = BatchCollator(tokenizer, max_length=max_length, padding=padding)
    dataloader_train = DataLoader(dataset_train, batch_size, collate_fn=batch_collator_train, drop_last=True)

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
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1)
    optimizer = AdamW(transformer.parameters(), lr=1, betas=(0.9, 0.997), eps=1e-9)
    scheduler = LambdaLR(optimizer, lambda steps: compute_lr(steps, transformer.d_model, warmup_steps))

    # Set accumulation steps to match the required number of tokens per batch if none was passed to the former
    if accumulation_steps == "None":
        actual_tokens_per_batch = batch_size * max_length
        if tokens_per_batch > actual_tokens_per_batch:
            accumulation_steps = math.ceil(tokens_per_batch / actual_tokens_per_batch)
        else:
            accumulation_steps = 1

    with tqdm(desc=f"{transformer.__class__.__name__} training", total=training_updates) as pbar:
        accumulation_loss = 0
        current_step = 0
        n_updates = 0
        training_end = False
        transformer.train()
        board = SummaryWriter()

        # Train loop
        while True:
            for batch in dataloader_train:
                # Retrieve encoder inputs, labels and decoder inputs
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                decoder_input_ids = batch["decoder_input_ids"].to(device)

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
                    board.add_scalar("Loss/train", mean_loss, current_step)
                    board.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], current_step)

                    # Check stopping criteria
                    if n_updates == training_updates:
                        training_end = True
                        break

                current_step += 1

            # End training if we reached the number of required updates
            if training_end:
                break

    # Close tensorboard
    board.flush()
    board.close()
