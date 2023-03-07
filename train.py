import torch
import yaml
from transformers import MBartTokenizer
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from translation_datasets import TranslationDataset
from models import Transformer
from utilities import model_size, model_n_parameters, generate_causal_mask, shift_tokens_right, compute_lr


def translation_batch_collate(collate_batch) -> (torch.Tensor, torch.Tensor):
    input_ids_batch = [sentence_pair["src_sentence"] for sentence_pair in collate_batch]
    labels_batch = [sentence_pair["tgt_sentence"] for sentence_pair in collate_batch]
    input_ids_batch = tokenizer(input_ids_batch, truncation=True, max_length=max_length, padding="longest",
                                return_tensors="pt")["input_ids"]
    labels_batch = tokenizer(text_target=labels_batch, truncation=True, max_length=max_length, padding="longest",
                             return_tensors="pt")["input_ids"]
    return {"input_ids": input_ids_batch, "labels": labels_batch}


if __name__ == "__main__":
    # Set-up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Retrieve configurations
    with open("config.yaml") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    huggingface_dataset = config["train_dataset"]
    batch_size = config["batch_size"]
    max_length = config["max_length"]
    warmup_steps = config["warmup_steps"]
    verbose = config["verbose_training"]
    log_steps = config["log_steps"]

    # Tokenizer
    tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained("tokenizers/mbart_tokenizer_cmlm", src_lang="en_XX",
                                                               tgt_lang="de_DE")
    print(f"Retrieved {tokenizer.__class__.__name__} with vocab size: {len(tokenizer)}")

    # Dataset
    dataset = load_dataset("yhavinga/ccmatrix", "en-de",
                           cache_dir="D:/MasterDegreeThesis/datasets/ccmatrix_en_de",
                           split="train[:4096]", ignore_verifications=True)

    dataset_train = TranslationDataset("en", "de", dataset, max_length, tokenizer=tokenizer, shift_labels_right=True)
    dataloader_train = DataLoader(dataset_train, batch_size, collate_fn=translation_batch_collate, drop_last=True)

    # Model
    transformer = Transformer(len(tokenizer)).to(device)
    n_parameters, n_trainable_parameters = model_n_parameters(transformer)
    transformer_size = model_size(transformer)
    print(f"Using {transformer.__class__.__name__} model")
    print(f"\tParameters: {n_parameters}\n\tTrainable parameters: {n_trainable_parameters}\n\tSize: {transformer_size}")

    # Useful token ids
    pad_token = tokenizer.pad_token_id
    src_lang_token = tokenizer.lang_code_to_id[tokenizer.src_lang]
    tgt_lang_token = tokenizer.lang_code_to_id[tokenizer.tgt_lang]
    eos_token = tokenizer.eos_token_id

    # Define loss function, optimizer and scheduler
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1)
    optimizer = AdamW(transformer.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lambda steps: compute_lr(steps, transformer.d_model, warmup_steps))

    # Train parameters
    current_step = 0
    epochs = 10
    total_loss = 0

    # Train loop
    transformer.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader_train):
            # Retrieve encoder inputs and labels, then create decoder inputs
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            decoder_input_ids = shift_tokens_right(labels, pad_token, tgt_lang_token).to(device)

            # Create masks
            e_pad_mask = (input_ids == pad_token).to(device)
            d_pad_mask = (decoder_input_ids == pad_token).to(device)
            d_mask = generate_causal_mask(decoder_input_ids.shape[-1]).to(device)

            # Compute predictions and loss
            logits = transformer(input_ids, decoder_input_ids, d_mask, e_pad_mask, d_pad_mask)
            loss = loss_fn(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))

            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if current_step % log_steps == 0:
                print(f"Epoch: {epoch}, Epoch step: {step}, Step: {current_step}, Loss: {loss.item()}")

            current_step += 1

        print(f"Epoch {epoch} ended at step {current_step}, Loss: {total_loss / len(dataset)}\n")
