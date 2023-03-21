import torch


SUPPORTED_LANGUAGES = {"ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX", "et": "et_EE",
                       "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN", "it": "it_IT", "ja": "ja_XX",
                       "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT", "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP",
                       "nl": "nl_XX", "ro": "ro_RO", "ru": "ru_RU", "si": "si_LK", "tr": "tr_TR", "vn": "vi_VN",
                       "zh": "zh_CN"}


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """
    Shift input ids one token to the right.
    :param input_ids: a tensor of shape (batch_size, seq_len) or (seq_len).
    :param pad_token_id: id of the pad token.
    :param decoder_start_token_id: start token id, which, in the case of the mBart tokenizer, is the target
        token language.
    :return: torch.Tensor shifted to the right.
    """
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)

    shifted_input_ids: torch.Tensor = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids = torch.where(shifted_input_ids == decoder_start_token_id, pad_token_id, shifted_input_ids)
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


def compute_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """
    Computes the current learning rate following the scheduling by Vaswani et al. https://arxiv.org/pdf/1706.03762.pdf.
    :param step: the current step.
    :param d_model: the model's embedding dimension.
    :param warmup_steps: the number of warmup steps
    :return: learning current for the current step.
    """
    if step == 0:
        step = 1

    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
