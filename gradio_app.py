import gradio
import torch
from transformers import MBartTokenizerFast

from continualnat.models import *
from continualnat.utils.utils import MBART_LANG_MAP


def translate_interface(model: str, training_type: str, translation_direction: str, src_text: str, device: str):
    device = "cuda:0" if device == "cuda" else device
    device = torch.device(device)

    # Change the tokenizer source and target language
    src_lang, tgt_lang = translation_direction.split(" -> ")
    src_lang_tokenizer = MBART_LANG_MAP[src_lang]
    tgt_lang_tokenizer = MBART_LANG_MAP[tgt_lang]
    tokenizer.src_lang = src_lang_tokenizer
    tokenizer.tgt_lang = tgt_lang_tokenizer

    # Load the model
    training_type = training_type.lower()
    if model == "Transformer":
        transformer_state_dict = torch.load(f"/disk1/a.ristori/models/Transformer_{training_type}")
        transformer_config = TransformerConfig(
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = Transformer(transformer_config)
        model.load_state_dict(transformer_state_dict)
    elif model == "CMLM":
        cmlm_state_dict = torch.load(f"/disk1/a.ristori/models/CMLM_{training_type}")
        cmlm_config = CMLMConfig(
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            length_token_id=None,
            pooler_size=256,
            glat_training=False,
        )
        model = CMLM(cmlm_config)
        model.load_state_dict(cmlm_state_dict)
    else:
        glat_state_dict = torch.load(f"/disk1/a.ristori/models/GLAT_{training_type}")
        glat_config = GLATConfig(
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            length_token_id=None,
            map_copy="soft",
            pooler_size=256,
        )
        model = GLAT(glat_config)
        model.load_state_dict(glat_state_dict)

    model.to(device)
    input_ids = tokenizer(src_text, truncation=True, max_length=128, padding=True, return_tensors="pt")["input_ids"]
    translation = model.generate(input_ids.to(device), tokenizer.lang_code_to_id[tgt_lang_tokenizer])
    if isinstance(model, CMLM):
        translation, _ = translation

    decoded_translation = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
    return decoded_translation


def launch_interface() -> None:
    with gradio.Blocks(title="ContinualNAT") as interface:
        with gradio.Row():
            model = gradio.Radio(["Transformer", "CMLM", "GLAT"], label="Model")
            training_type = gradio.Radio(["Joint", "Incremental", "Replay"], label="Training")
            translation_direction = gradio.Radio(
                ["en -> de", "de -> en", "en -> fr", "fr -> en", "en -> es", "es -> en"],
                label="Translation direction",
            )
            device = gradio.Radio(["cpu", "cuda"], label="Device")

        with gradio.Row():
            src_text = gradio.Textbox(label="Source sentence", placeholder="Write your source sentence here")
            tgt_text = gradio.Textbox(label="Translation")

        translate = gradio.Button("Translate")
        translate.click(translate_interface, [model, training_type, translation_direction, src_text, device], tgt_text)
        examples = gradio.Examples(
            [
                "What are you doing for the session?",
                "That was incredible, how did you do it?",
                "I went to see a friend of mine yesterday, we had a great time together.",
            ],
            src_text,
        )

    interface.launch(debug=True)


if __name__ == "__main__":
    tokenizer = MBartTokenizerFast(tokenizer_file="tokenizers/sp_32k.json", model_max_length=1024, cls_token="<length>")
    launch_interface()
