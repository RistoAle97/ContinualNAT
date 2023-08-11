import torch
from datasets import load_dataset
from transformers import AutoTokenizer, MarianTokenizer

from src.data import distill_dataset, push_distilled_dataset_to_hub


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ccmatrix_to_distill = load_dataset("yhavinga/ccmatrix", "de-en", split="train[:1000000]",
                                       cache_dir="/disk1/a.ristori/datasets/ccmatrix", verification_mode="no_checks")
    marian_tokenizer: MarianTokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    distill_dataset("ct2-opus-mt-en-de", marian_tokenizer, ccmatrix_to_distill, "ccmatrix", "en", "de", device, 4, 4096,
                    save_dir="/disk1/a.ristori/datasets/distillation")
    push_distilled_dataset_to_hub("/disk1/a.ristori/datasets/distilled_ccmatrix_to_push",
                                  "thesistranslation/distilled_ccmatrix_en_de", "en", "de",
                                  "/disk1/a.ristori/datasets/distillation/distilled_ccmatrix.en_de")
    distilled_ccmatrix = load_dataset("thesistranslation/distilled_ccmatrix_en_de",
                                      cache_dir="/disk1/a.ristori/datasets/distilled_ccmatrix",
                                      verification_mode="no_checks")
