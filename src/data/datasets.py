import datasets
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from tqdm.auto import tqdm
from typing import Dict, Iterator, Set, Union


class TranslationDatasetCore:

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: Union[datasets.Dataset, datasets.IterableDataset],
                 skip_idxs: Set[int] = None) -> None:
        """
        Base class for all the translation datasets.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the Huggingface dataset to wrap.
        """
        # Source and target languages
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        if "translation" not in dataset.features.keys():
            raise ValueError("You should use a dataset suitable for the translation task.")

        # Dataset and stats about it
        self.dataset = dataset
        self.avg_length_src = 0
        self.avg_length_tgt = 0
        self.max_length_src = 0
        self.max_length_tgt = 0

        # Dataset's indexes that should be skipped (duplicated, corrupted or unwanted sentences)
        self.skip_idxs = set() if skip_idxs is None else skip_idxs

    def compute_stats(self) -> Dict[str, Union[int, float]]:
        """
        Computes and updates the dataset's stats.
        :return: a dict containing the average and max lengths for both source and target languages.
        """
        for sample in tqdm(self.dataset, "Computing average and max length for source and target"):
            sample: dict
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


class TranslationDataset(TranslationDatasetCore, Dataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.Dataset,
                 skip_idxs: Set[int] = None) -> None:
        """
        Translation dataset defined by source and target languages.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the Huggingface dataset to wrap.
        """
        TranslationDatasetCore.__init__(self, src_lang, tgt_lang, dataset, skip_idxs)
        Dataset.__init__(self)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, str]:
        while idx in self.skip_idxs:
            idx = np.random.randint(0, self.__len__())

        sentence_pair = self.dataset[idx]["translation"]
        src_sentence = sentence_pair[self.src_lang]
        tgt_sentence = sentence_pair[self.tgt_lang]
        return {"src_sentence": src_sentence, "tgt_sentence": tgt_sentence}


class IterableTranslationDataset(TranslationDatasetCore, IterableDataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 dataset: datasets.IterableDataset) -> None:
        """
        Translation dataset for iterable Hugginface datasets, defined by source and target languages.
        :param src_lang: the source language.
        :param tgt_lang: the target language.
        :param dataset: the iterable Hugginface dataset to wrap.
        """
        TranslationDatasetCore.__init__(self, src_lang, tgt_lang, dataset)
        IterableDataset.__init__(self)

    def __iter__(self) -> Iterator:
        for sentence_pair_langs in self.dataset:
            if "id" in sentence_pair_langs.keys():
                sentence_pair_id = sentence_pair_langs["id"]
                if sentence_pair_id in self.skip_idxs:
                    continue

            sentence_pair = sentence_pair_langs["translation"]
            src_sentence = sentence_pair[self.src_lang]
            tgt_sentence = sentence_pair[self.tgt_lang]
            yield {"src_sentence": src_sentence, "tgt_sentence": tgt_sentence}


class TextDataset(Dataset):

    def __init__(self, dataset: datasets.Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, str]:
        tgt_sentence = self.dataset[idx]["text"]
        return {"tgt_sentence": tgt_sentence}
