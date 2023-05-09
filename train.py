import torch
from transformers import MBartTokenizerFast
from src.models import *
from src import MultilingualTrainer


if __name__ == "__main__":
    # Set-up
    torch.set_float32_matmul_precision("medium")

    # Translation directions
    translation_directions = ["en-es", "es-en", "en-fr", "fr-en", "de-en", "en-de"]

    # Tokenizer
    tokenizer = MBartTokenizerFast(tokenizer_file="tokenizers/sp_32k.json", model_max_length=1024, cls_token="<length>")
    print(f"Using {tokenizer.__class__.__name__} with vocab size: {len(tokenizer)}")

    # Model
    model_config = CMLMConfig(len(tokenizer), sos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
                              length_token_id=tokenizer.cls_token_id)
    model = CMLM(model_config)
    '''model_config = TransformerConfig(len(tokenizer), **model_parameters, sos_token_id=tokenizer.bos_token_id,
                                     eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = Transformer(model_config)'''

    # Trainer
    trainer = MultilingualTrainer(tokenizer=tokenizer, max_length=128, padding="longest", use_cls_token=True,
                                  train_dataset_cache_dir="/disk1/a.ristori/datasets/ccmatrix",
                                  val_dataset_cache_dir="/disk1/a.ristori/datasets/flores200",
                                  nmt_directions=translation_directions, train_bsz=512, val_bsz=128,
                                  tokens_per_batch=128000, train_steps=100000, val_every_n_steps=10000,
                                  log_every_n_steps=500, ckpt_every_n_steps=10000, streaming=False)

    # Train the model
    trainer.train(model)
