import torch
from transformers import MBartTokenizer
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from translation_datasets import TranslationDataset, translation_collate_fn
from models import Transformer
from schedulers import TransformerScheduler
from utilities import model_size, model_n_parameters, generate_causal_mask, shift_tokens_right


if __name__ == "__main__":
    # Set-up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Tokenizer
    tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained("tokenizers/mbart_tokenizer_cmlm", src_lang="en_XX",
                                                               tgt_lang="de_DE")
    print(f"Retrieved {tokenizer.__class__} with vocab size: {len(tokenizer)}")

    # Dataset
    dataset = load_dataset("yhavinga/ccmatrix", "en-de",
                           cache_dir="D:/MasterDegreeThesis/datasets/ccmatrix_en_de",
                           split="train[:1000]", ignore_verifications=True)

    dataset_train = TranslationDataset("en", "de", dataset, tokenizer=tokenizer)
    dataloader_train = DataLoader(dataset_train, 8, collate_fn=translation_collate_fn, drop_last=True)
    print(f"Built en-de dataset")

    # Model
    transformer = Transformer(len(tokenizer)).to(device)
    n_parameters, n_trainable_parameters = model_n_parameters(transformer)
    transformer_size = model_size(transformer)
    print("Using transformer base model")
    print(f"\tParameters: {n_parameters}\n\tTrainable parameters: {n_trainable_parameters}\n\tSize: {transformer_size}")

    # Useful token ids
    pad_token = tokenizer.pad_token_id
    src_lang_token = tokenizer.lang_code_to_id[tokenizer.src_lang]
    tgt_lang_token = tokenizer.lang_code_to_id[tokenizer.tgt_lang]
    eos_token = tokenizer.eos_token_id

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1)
    optimizer = AdamW(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerScheduler(optimizer, transformer.d_model, 4000)

    # Set the model in training mode
    transformer.train()
    total_loss = 0

    # Train loop
    for epoch in range(10):
        for step, batch in enumerate(tqdm(dataloader_train)):
            # Retrieve encoder inputs and labels, then create decoder inputs
            input_ids, labels = batch.to(device)
            decoder_input_ids = shift_tokens_right(labels, pad_token, tgt_lang_token)

            # Create masks
            e_pad_mask = (input_ids == pad_token).to(device)
            d_pad_mask = (decoder_input_ids == pad_token).to(device)
            d_mask = generate_causal_mask(decoder_input_ids.shape[-1]).to(device)

            # Compute predictions and loss
            logits = transformer(input_ids, decoder_input_ids, d_mask, e_pad_mask, d_pad_mask)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            total_loss += loss

            # Update weights and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    total_loss /= len(dataloader_train)
