import datasets
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from typing import Dict, Union


class TranslationDataset(Dataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.Dataset) -> None:
        """
        Translation dataset defined by the source and target languages.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the Huggingface dataset to wrap.
        """
        super().__init__()
        # Source and target languages
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Dataset and stats about it
        self.dataset = dataset
        self.avg_length_src = 0
        self.avg_length_tgt = 0
        self.max_length_src = 0
        self.max_length_tgt = 0

    def compute_stats(self) -> Dict[str, Union[int, float]]:
        """
        Computes and updates the dataset's stats.
        :return: a dict containing the average and max lenght for both source and target languages.
        """
        for sample in tqdm(self.dataset, "Computing average and max length for source and target"):
            sentences = sample["translation"]
            src_sentence: str = sentences[self.src_lang]
            tgt_sentence: str = sentences[self.tgt_lang]
            length_splitted_src = len(src_sentence.split())
            length_splitted_tgt = len(tgt_sentence.split())
            self.max_length_src = max(self.max_length_src, length_splitted_src)
            self.max_length_tgt = max(self.max_length_tgt, length_splitted_tgt)
            self.avg_length_src += length_splitted_src
            self.avg_length_tgt += length_splitted_tgt

        self.avg_length_src /= len(self.dataset)
        self.avg_length_tgt /= len(self.dataset)
        return {"max_length_src": self.max_length_src, "max_length_tgt": self.max_length_tgt,
                "avg_length_src": self.avg_length_src, "avg_length_tgt": self.avg_length_tgt}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, str]:
        sentence_pair = self.dataset[idx]["translation"]
        src_sentence = sentence_pair[self.src_lang]
        tgt_sentence = sentence_pair[self.tgt_lang]
        return {"src_sentence": src_sentence, "tgt_sentence": tgt_sentence}


class TranslationDatasetCMLM(TranslationDataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.Dataset) -> None:
        """
        Simple variation of the standard translation dataset mainly used for masked language models like BERT or CMLM,
        a <length> token is placed at the start of each source sentence.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the Huggingface dataset to wrap.
        """
        super().__init__(src_lang, tgt_lang, dataset)

    def __getitem__(self, idx) -> Dict[str, str]:
        sentence_pair = super().__getitem__(idx)
        return {"src_sentence": "<length> " + sentence_pair["src_sentence"],
                "tgt_sentence": sentence_pair["tgt_sentence"]}
