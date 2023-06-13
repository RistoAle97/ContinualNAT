import evaluate
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, MBartTokenizer, MBartTokenizerFast
from tqdm.auto import tqdm
from src.data import *
from src.models import TransformerCore, CMLM


def compute_sacrebleu(model: TransformerCore,
                      dataset: TranslationDataset,
                      tokenizer: PreTrainedTokenizerBase,
                      bsz: int = 32,
                      metric_tokenize: str = "13a"):
    if metric_tokenize not in ["none", "zh", "13a", "intl", "char", "ja-mecab"]:
        raise ValueError("Wrong tokenizer passed for the SacreBLEU metric, use one of the following: \"none\", \"zh\", "
                         "\"13a\", \"char\", \"ja-mecab\".")
    
    scb = evaluate.load("sacrebleu")
    device = model.device
    if isinstance(model, CMLM):
        batch_collator = BatchCollatorCMLM(model.pad_token_id, model.mask_token_id)
    else:
        shift_lang_token = True if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)) else False
        batch_collator = BatchCollator(shift_lang_token=shift_lang_token, pad_token_id=model.pad_token_id)

    dataloader = DataLoader(dataset, batch_size=bsz, collate_fn=batch_collator)
    translations = []
    targets = []
    for i, batch in enumerate(tqdm(dataloader)):
        translation = model.generate(batch["input_ids"].to(device), tokenizer.lang_code_to_id["de_DE"])
        decoded_translation = tokenizer.batch_decode(translation, skip_special_tokens=True)
        translations.extend([translation for translation in decoded_translation])
        targets.extend(batch["references"])

    return scb.compute(predictions=translations, references=targets, tokenize=metric_tokenize)
