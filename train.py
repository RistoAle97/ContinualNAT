import os

import torch
import yaml
from datasets import load_dataset
from transformers import MBartTokenizerFast

from continualnat.data import *
from continualnat.models import *
from continualnat.metrics import compute_sacrebleu
from continualnat.trainers import *

if __name__ == "__main__":
    logger_version_train = "Model_continual_exp0"
    save_model_directory_empty = os.path.join("/disk1/a.ristori/models/", logger_version_train)
    save_model_directory = logger_version_train
    # Set-up
    torch.set_float32_matmul_precision("medium")

    # Keep track of the wmt14 test set duplicates inside the ccmatrix datasets
    with open("duplicates.yaml") as duplicates_file:
        duplicates = yaml.load(duplicates_file, Loader=yaml.FullLoader)

    en_de_duplicates = set(duplicates["duplicates_en_de"] + duplicates["duplicates_de_en"])
    en_fr_duplicates = set(duplicates["duplicates_en_fr"] + duplicates["duplicates_fr_en"])
    en_es_duplicates = set(duplicates["duplicates_en_es"] + duplicates["duplicates_es_en"])
    duplicates = {"en-de": en_de_duplicates, "en-fr": en_fr_duplicates, "en-es": en_es_duplicates}

    # Set up the Tokenizer
    tokenizer = MBartTokenizerFast(tokenizer_file="tokenizers/sp_32k.json", model_max_length=1024, cls_token="<length>")
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    print(f"Using {tokenizer.__class__.__name__} with vocab size: {len(tokenizer)}")

    # Load the datasets used as validation and test sets from the huggingface hub
    wmt_en_de = load_dataset(
        "thesistranslation/wmt14", "de-en", cache_dir="/disk1/a.ristori/datasets/wmt14", verification_mode="no_checks"
    )
    wmt_en_fr = load_dataset(
        "thesistranslation/wmt14", "fr-en", cache_dir="/disk1/a.ristori/datasets/wmt14", verification_mode="no_checks"
    )
    wmt_en_es = load_dataset(
        "thesistranslation/wmt14", "es-en", cache_dir="/disk1/a.ristori/datasets/wmt14", verification_mode="no_checks"
    )
    wmt_datasets = {"en-de": wmt_en_de, "en-fr": wmt_en_fr, "en-es": wmt_en_es}

    # Build the model
    '''model_config = CMLMConfig(
        vocab_size=len(tokenizer),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        length_token_id=None,
        pooler_size=256,
        glat_training=False,
    )
    model = CMLM(model_config)'''
    model_config = TransformerConfig(
        vocab_size=len(tokenizer),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    model = Transformer(model_config)
    '''model_config = GLATConfig(
        vocab_size=len(tokenizer),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        length_token_id=None,
        map_copy="soft",
        pooler_size=256,
    )
    model = GLAT(model_config)'''

    # Check whether the model is using the length token
    use_cls_token = False
    if hasattr(model, "length_token_id") and model.length_token_id is not None:
        use_cls_token = True

    # Build the train, validation and test datasets
    train_datasets = []
    val_datasets = []
    test_datasets = []
    exps_lang_pairs = [["en-de", "de-en"], ["en-fr", "fr-en"], ["en-es", "es-en"]]
    # exps_lang_pairs = [["en-de"]]
    train_exps = []
    val_exps = []
    lang_tokens_only_encoder = True if isinstance(model, GLAT) else False
    for exp_lang_pair in exps_lang_pairs:
        train_current_exp = []
        val_current_exp = []
        for lang_pair in exp_lang_pair:
            src_lang, tgt_lang = lang_pair.split("-")
            lang_pair_key = lang_pair if src_lang == "en" else f"{tgt_lang}-{src_lang}"
            dataset_duplicates = duplicates[lang_pair_key]
            wmt_dataset = wmt_datasets[lang_pair_key]
            distilled_ccmatrix = load_dataset(
                path=f"thesistranslation/distilled-ccmatrix-{src_lang}-{tgt_lang}",
                split="train",
                cache_dir="/disk1/a.ristori/datasets/distilled_ccmatrix",
                verification_mode="no_checks",
            )
            train_dataset = TranslationDataset(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                dataset=distilled_ccmatrix,
                tokenizer=tokenizer,
                max_length=128,
                use_cls_token=use_cls_token,
                skip_idxs=dataset_duplicates,
                lang_tokens_only_encoder=lang_tokens_only_encoder,
            )
            val_dataset = TranslationDataset(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                dataset=wmt_dataset["validation"],
                tokenizer=tokenizer,
                max_length=128,
                use_cls_token=use_cls_token,
                lang_tokens_only_encoder=lang_tokens_only_encoder,
            )
            test_dataset = TranslationDataset(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                dataset=wmt_dataset["test"],
                tokenizer=tokenizer,
                max_length=128,
                use_cls_token=use_cls_token,
                lang_tokens_only_encoder=lang_tokens_only_encoder,
            )
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
            train_current_exp.append(train_dataset)
            val_current_exp.append(val_dataset)

        train_exps.append(train_current_exp)
        val_exps.append(val_current_exp)

    # Set up the trainer
    '''trainer = MultilingualTrainer(
        tokenizer=tokenizer,
        train_steps=300000,
        val_every_n_steps=10000,
        log_every_n_steps=500,
        dataloader_num_workers=8,
        log_directory="/disk1/a.ristori/",
        use_wandb=True,
    )'''
    buffer_size = sum([len(train_dataset) for train_dataset in train_datasets]) * 5 // 100
    continual_trainer = ContinualTrainer(
        tokenizer=tokenizer,
        buffer_size=buffer_size,
        train_steps=200000,
        val_every_n_steps=10000,
        log_every_n_steps=500,
        dataloader_num_workers=8,
        log_directory="/disk1/a.ristori/",
        use_wandb=True,
    )

    # Train the model
    version = "Transformer_continual"
    '''trainer.train(
        model=model,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        train_bsz=512,
        val_bsz=32,
        tokens_per_batch=128000,
        logger_version=version,
        wandb_project="Thesis",
    )'''
    continual_trainer.train(
        model=model,
        exps=train_exps,
        val_datasets=val_exps,
        train_bsz=512,
        val_bsz=32,
        tokens_per_batch=128000,
        logger_version=version,
        save_model_each_exp=True,
        wandb_project="Thesis",
    )

    # Save the model
    torch.save(model.state_dict(), f"/disk1/a.ristori/models/{version}")

    # Evaluate the model by computing the SacreBLEU score for each translation direction
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for test_dataset in test_datasets:
        test_bleu = compute_sacrebleu(model, test_dataset, tokenizer, 32, True, metric_tokenize={"13a", "intl", "spm"})
        print(
            f"{test_dataset.src_lang}-{test_dataset.tgt_lang} BLEU scores\n"
            f"13a: {test_bleu['13a']}\n"
            f"intl: {test_bleu['intl']}\n"
        )
