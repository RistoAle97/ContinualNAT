import math
from typing import List, Tuple

import matplotlib as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

# Maps iso codes into language codes for the Mbart tokenizer
MBART_LANG_MAP = {"ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX", "et": "et_EE",
                  "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN", "it": "it_IT", "ja": "ja_XX",
                  "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT", "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP",
                  "nl": "nl_XX", "ro": "ro_RO", "ru": "ru_RU", "si": "si_LK", "tr": "tr_TR", "vn": "vi_VN",
                  "zh": "zh_CN"}

# Maps ISO codes into language codes for the nllb and flores-200 datasets
NLLB_FLORES200_LANG_MAP = {"en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn"}


def shift_lang_token_right(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Shift input ids one token to the right by moving the language token to the sequence's start in a MBart style.
    :param input_ids: a tensor of shape (1, seq_len) or (seq_len).
    :param pad_token_id: id of the pad token.
    :return: torch.Tensor with the language token moved to the beginning of each sequence.
    """
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)

    batch_size, seq_len = input_ids.size()
    shifted_input_ids: torch.Tensor = input_ids.clone()

    # Compute the indexes of the lang tokens and retrieve them
    eos_idxs = (input_ids.ne(pad_token_id).sum(dim=1) - 1).view(-1)
    eos_idxs += torch.arange(0, batch_size * seq_len, seq_len)
    decoder_start_token_ids = shifted_input_ids.view(-1).gather(0, eos_idxs).squeeze(0)

    # Pad the previous positions where the language tokens have been found
    shifted_input_ids.view(-1)[eos_idxs] = pad_token_id

    # Shift the tokens to the right and put the language tokens at the start
    shifted_input_ids[:, 1:] = shifted_input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_ids
    return shifted_input_ids


def compute_accumulation_steps(batch_size: int, max_length: int, tokens_per_batch: int) -> int:
    """
    Computes the number of accumulation steps needed to reach, at least, the tokens per batch wanted by the user given
    the batch size and the maximum number of tokens allowed per sentence.
    :param batch_size: the batch size or, more easily explained, the number of sentences per batch.
    :param max_length: the maximum number of tokens allowed per sentence, this also includes special tokens such as the
        language ones.
    :param tokens_per_batch: the number of tokens that the model should see at each training step.
    :return: the number of accumulation steps needed to reach the desired tokens per batch.
    """
    actual_tokens_per_batch = batch_size * max_length
    if tokens_per_batch > actual_tokens_per_batch:
        accumulation_steps = math.ceil(tokens_per_batch / actual_tokens_per_batch)
    else:
        accumulation_steps = 1

    return accumulation_steps


def plot_lr_scheduler(lr_scheduler: LambdaLR, num_steps: int = 100000) -> None:
    """
    Plot the learning reate scheduler steps.
    :param lr_scheduler: the learning rate scheduler.
    :param num_steps: the number of steps to consider (default=100000).
    """
    lrs = []
    for _ in range(1, num_steps):
        lr_scheduler.optimizer.step()
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_last_lr())

    scheduler_steps = np.arange(len(lrs))
    plt.plot(scheduler_steps, lrs, linewidth=2)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Learning rate", fontsize=14)
    plt.title("Learning rate schedule", fontsize=19)
    plt.legend(["Eta"], fontsize=14)
    plt.grid()
    plt.show()


def compute_repeated_tokens(translations_tokens: List[str]) -> Tuple[int, float]:
    """
    Compute the percentage of repeated tokens for a generated translation.
    :param translations_tokens: a list of tokens that should not have any special token in order to have a correct
        estimate.
    :return: a tuple containing the number and percentage of repeated tokens.
    """
    repeated_tokens = 0
    for i, token in enumerate(translations_tokens):
        if i == 0:
            continue

        if token == translations_tokens[i - 1]:
            repeated_tokens += 1

    return repeated_tokens, repeated_tokens / len(translations_tokens)
