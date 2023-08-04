import torch
import yaml
from torch.optim import AdamW
from datasets import load_dataset
from transformers import MBartTokenizerFast

from src.data import *
from src.models import *
from src.metrics import compute_sacrebleu
from src import MultilingualTrainer


if __name__ == "__main__":
    # Set-up
    torch.set_float32_matmul_precision("medium")

    # Keep track of the wmt14 test set duplicates inside the ccmatrix datasets
    with open("duplicates.yaml") as duplicates_file:
        duplicates = yaml.load(duplicates_file, Loader=yaml.FullLoader)

    en_de_duplicates = set(duplicates["duplicates_en_de"] + duplicates["duplicates_de_en"])
    en_fr_duplicates = set(duplicates["duplicates_en_fr"] + duplicates["duplicates_fr_en"])
    en_es_duplicates = set(duplicates["duplicates_en_es"] + duplicates["duplicates_es_en"])

    # Set up the Tokenizer
    tokenizer = MBartTokenizerFast(tokenizer_file="tokenizers/sp_32k.json", model_max_length=1024, cls_token="<length>")
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    print(f"Using {tokenizer.__class__.__name__} with vocab size: {len(tokenizer)}")

    # Load the datasets from the huggingface hub
    wmt_en_de = load_dataset("wmt14", "de-en", cache_dir="/disk1/a.ristori/datasets/wmt14",
                             verification_mode="no_checks")
    wmt_en_fr = load_dataset("wmt14", "fr-en", cache_dir="/disk1/a.ristori/datasets/wmt14",
                             verification_mode="no_checks")
    wmt_en_es = load_dataset("thesistranslation/wmt14", "es-en", cache_dir="/disk1/a.ristori/datasets/wmt14",
                             verification_mode="no_checks")
    ccmatrix_en_de = load_dataset("yhavinga/ccmatrix", "de-en", cache_dir="/disk1/a.ristori/datasets/ccmatrix",
                                  verification_mode="no_checks")
    ccmatrix_en_fr = load_dataset("yhavinga/ccmatrix", "en-fr", cache_dir="/disk1/a.ristori/datasets/ccmatrix",
                                  verification_mode="no_checks")
    ccmatrix_en_es = load_dataset("yhavinga/ccmatrix", "en-es", cache_dir="/disk1/a.ristori/datasets/ccmatrix",
                                  verification_mode="no_checks")

    # Model
    # model_config = CMLMConfig(len(tokenizer), bos_token_id=bos_token_id, eos_token_id=eos_token_id,
    #                           pad_token_id=pad_token_id, mask_token_id=mask_token_id, length_token_id=None,
    #                           pooler_size=256)
    # model = CMLM(model_config)
    model_config = TransformerConfig(len(tokenizer), bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                     pad_token_id=pad_token_id)
    model = Transformer(model_config)

    if isinstance(model, CMLM):
        optimizer = AdamW(model.parameters(), lr=5e-4, eps=1e-6)
        model.change_optimizer(optimizer)

    use_cls_token = False
    if hasattr(model, "length_token_id") and model.length_token_id is not None:
        use_cls_token = True

    # Build the train datasets
    train_en_de = TranslationDataset("en", "de", ccmatrix_en_de["train"], tokenizer, max_length=128,
                                     use_cls_token=use_cls_token, skip_idxs=en_de_duplicates, fill_to_max_length=False)
    train_de_en = TranslationDataset("de", "en", ccmatrix_en_de["train"], tokenizer, max_length=128,
                                     use_cls_token=use_cls_token, skip_idxs=en_de_duplicates, fill_to_max_length=False)
    train_en_fr = TranslationDataset("en", "fr", ccmatrix_en_fr["train"], tokenizer, max_length=128,
                                     use_cls_token=use_cls_token, skip_idxs=en_fr_duplicates, fill_to_max_length=False)
    train_fr_en = TranslationDataset("fr", "en", ccmatrix_en_fr["train"], tokenizer, max_length=128,
                                     use_cls_token=use_cls_token, skip_idxs=en_fr_duplicates, fill_to_max_length=False)
    train_en_es = TranslationDataset("en", "es", ccmatrix_en_es["train"], tokenizer, max_length=128,
                                     use_cls_token=use_cls_token, skip_idxs=en_es_duplicates, fill_to_max_length=False)
    train_es_en = TranslationDataset("es", "en", ccmatrix_en_es["train"], tokenizer, max_length=128,
                                     use_cls_token=use_cls_token, skip_idxs=en_es_duplicates, fill_to_max_length=False)
    train_datasets = [train_en_de, train_de_en, train_en_fr, train_fr_en, train_en_es, train_es_en]

    # Build the validation datasets
    val_en_de = TranslationDataset("en", "de", wmt_en_de["validation"], tokenizer, max_length=128,
                                   use_cls_token=use_cls_token)
    val_de_en = TranslationDataset("de", "en", wmt_en_de["validation"], tokenizer, max_length=128,
                                   use_cls_token=use_cls_token)
    val_en_fr = TranslationDataset("en", "fr", wmt_en_fr["validation"], tokenizer, max_length=128,
                                   use_cls_token=use_cls_token)
    val_fr_en = TranslationDataset("fr", "en", wmt_en_fr["validation"], tokenizer, max_length=128,
                                   use_cls_token=use_cls_token)
    val_en_es = TranslationDataset("en", "es", wmt_en_es["validation"], tokenizer, max_length=128,
                                   use_cls_token=use_cls_token)
    val_es_en = TranslationDataset("es", "en", wmt_en_es["validation"], tokenizer, max_length=128,
                                   use_cls_token=use_cls_token)
    val_datasets = [val_en_de, val_de_en, val_en_fr, val_fr_en, val_en_es, val_es_en]

    # Set up the trainer
    trainer = MultilingualTrainer(tokenizer=tokenizer, train_steps=200000, val_every_n_steps=10000,
                                  log_every_n_steps=500, ckpt_every_n_steps=10000, log_directory="/disk1/a.ristori/",
                                  use_wandb=True)

    # Train the model
    version = "Transformer_ccmatrix_bsz256"
    trainer.train(model, train_datasets, val_datasets, train_bsz=256, val_bsz=32, tokens_per_batch=128000,
                  logger_version=version)

    # Save the model
    torch.save(model.state_dict(), f"/disk1/a.ristori/models/{version}")

    # Evaluate the model by computing the SacreBLEU score for each translation direction
    test_en_de = TranslationDataset("en", "de", wmt_en_de["test"], tokenizer, max_length=128,
                                    use_cls_token=use_cls_token)
    test_de_en = TranslationDataset("de", "en", wmt_en_de["test"], tokenizer, max_length=128,
                                    use_cls_token=use_cls_token)
    test_en_fr = TranslationDataset("en", "fr", wmt_en_fr["test"], tokenizer, max_length=128,
                                    use_cls_token=use_cls_token)
    test_fr_en = TranslationDataset("fr", "en", wmt_en_fr["test"], tokenizer, max_length=128,
                                    use_cls_token=use_cls_token)
    test_en_es = TranslationDataset("en", "es", wmt_en_es["test"], tokenizer, max_length=128,
                                    use_cls_token=use_cls_token)
    test_es_en = TranslationDataset("es", "en", wmt_en_es["test"], tokenizer, max_length=128,
                                    use_cls_token=use_cls_token)
    test_datasets = [test_en_de, test_de_en, test_en_fr, test_fr_en, test_es_en, test_es_en]
    for test_dataset in test_datasets:
        test_bleu = compute_sacrebleu(model, test_dataset, tokenizer, 32, True, {"13a", "intl"})
        print(f"{test_dataset.src_lang}-{test_dataset.tgt_lang} BLEU scores\n"
              f"13a: {test_bleu['13a']}\n"
              f"intl: {test_bleu['intl']}\n")
