import torch
from torch.optim import AdamW
from transformers import MBartTokenizerFast
from src.models import *
from src import MultilingualTrainer


if __name__ == "__main__":
    # Set-up
    torch.set_float32_matmul_precision("medium")

    # Translation directions
    translation_directions = ["en-de", "de-en", "en-fr", "fr-en"]

    # Tokenizer
    tokenizer = MBartTokenizerFast(tokenizer_file="tokenizers/sp_32k.json", model_max_length=1024, cls_token="<length>")
    print(f"Using {tokenizer.__class__.__name__} with vocab size: {len(tokenizer)}")

    # Model
    '''model_config = CMLMConfig(len(tokenizer), sos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
                              length_token_id=tokenizer.cls_token_id, dropout=0.3, label_smoothing=0.1)
    model = CMLM(model_config)
    optimizer = AdamW(model.parameters(), lr=5e-4, eps=1e-6)
    model.change_optimizer(optimizer)
    model.change_lr_scheduler("inverse_sqrt")'''
    model_config = TransformerConfig(len(tokenizer), sos_token_id=tokenizer.bos_token_id,
                                     eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                     label_smoothing=0.1)
    model = Transformer(model_config)
    model.change_lr_scheduler("inverse_sqrt")

    # Trainer
    trainer = MultilingualTrainer(tokenizer=tokenizer, max_length=128, padding="longest", use_cls_token=True,
                                  train_dataset_cache_dir="/disk1/a.ristori/datasets/ccmatrix",
                                  val_dataset_cache_dir="/disk1/a.ristori/datasets/wmt14",
                                  nmt_directions=translation_directions, train_bsz=512, val_bsz=32,
                                  tokens_per_batch=128000, train_steps=100000, val_every_n_steps=10000,
                                  log_every_n_steps=500, ckpt_every_n_steps=10000, streaming=False)

    # Train the model
    trainer.train(model)
