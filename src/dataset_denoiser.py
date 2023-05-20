import datasets
from multiprocessing import cpu_count
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, Future
from sacremoses import MosesTokenizer
from langdetect import DetectorFactory, detect
from typing import Dict, List


class DatasetDenoiser:

    def __init__(self, n_threads: int = 1, length_threshold: int = 3, length_ratio: int = 2) -> None:
        """
        A simple translation dataset denoiser, it will save the id of those pairs whose source or target sentences that
        are empty, too short, to different while tokenized or their language is wrong.
        :param n_threads: the number of threads that the denoiser will use (default=1).
        :param length_threshold: the threshold for the sentences' length (default=3)
        :param length_ratio: the ratio between the source and target sentences for which they will be considered
            too different (default=2).
        """
        DetectorFactory.seed = 0
        max_threads = cpu_count()
        self.n_threads = n_threads if n_threads <= max_threads else max_threads
        self.length_threshold = length_threshold
        self.length_ratio = length_ratio

    def __compute_noisy_sentences(self,
                                  dataset: dict[str, List[Dict[str, str]]],
                                  i_s: int,
                                  progr_pbar: tqdm,
                                  src_lang: str,
                                  tgt_lang: str) -> Dict[str, List[int]]:
        full_stop = []
        empty = []
        too_short = []
        too_different = []
        wrong_lang = []
        stops = [" ", ".", "," ";", ":"]
        sentence_pairs = dataset["translation"]

        # Tokenizers for both source and target languages
        src_tokenizer = MosesTokenizer(src_lang)
        tgt_tokenizer = MosesTokenizer(tgt_lang)
        for lang_pair in sentence_pairs:
            src_sentence: str = lang_pair[src_lang]
            tgt_sentence: str = lang_pair[tgt_lang]

            if src_sentence in stops or tgt_sentence in stops:
                full_stop.append(i_s)
                progr_pbar.update(1)
                i_s += 1
                continue

            if src_sentence == "" or tgt_sentence == "":
                empty.append(i_s)
                progr_pbar.update(1)
                i_s += 1
                continue

            src_detected_lang: str = detect(src_sentence)
            tgt_detected_lang: str = detect(tgt_sentence)
            if src_lang != src_detected_lang or tgt_lang != tgt_detected_lang:
                wrong_lang.append(i_s)
                progr_pbar.update(1)
                i_s += 1
                continue

            src_tokens = src_tokenizer.tokenize(src_sentence)
            tgt_tokens = tgt_tokenizer.tokenize(tgt_sentence)
            if len(src_tokens) >= self.length_threshold or len(tgt_tokens) >= self.length_threshold:
                if len(src_tokens) / len(tgt_tokens) > self.length_ratio:
                    too_different.append(i_s)

                if len(src_tokens) / len(tgt_tokens) > self.length_ratio:
                    too_different.append(i_s)
            else:
                too_short.append(i_s)

            progr_pbar.update(1)
            i_s += 1

        return {"full_stop": full_stop, "too_short": too_short, "empty": empty, "wrong_lang": wrong_lang,
                "too_different": too_different}

    @staticmethod
    def __write_noisy_sentences_on_file(file: str,
                                        noisy_sentences: List[int],
                                        dataset: datasets.Dataset,
                                        src_lang: str,
                                        tgt_lang: str) -> None:
        if "translation" not in dataset.features.keys():
            raise ValueError("Only translation datasets are accepted")

        with open(file, "w") as datafile:
            for i in noisy_sentences:
                lang_pair = dataset[i]["translation"]
                src_sentence = lang_pair[src_lang]
                tgt_sentence = lang_pair[tgt_lang]
                datafile.write(f"{i}\n{src_lang}: {src_sentence}\n{tgt_lang}: {tgt_sentence}\n\n")

    def denoise(self, dataset: datasets.Dataset, src_lang: str, tgt_lang: str):
        """
        Denoises a dataset given the source and target languages.
        :param dataset: a Huggingface Dataset.
        :param src_lang: the souce language.
        :param tgt_lang: the target language.
        """
        dataset_name = f"{dataset.builder_name}_{src_lang}_{tgt_lang}"
        sentence_pairs_per_thread, reminder = divmod(len(dataset), self.n_threads)
        i_start, i_end = 0, sentence_pairs_per_thread
        futures: List[Future] = []
        with tqdm(total=len(dataset)) as pbar:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                for thr in range(self.n_threads):
                    futures.append(executor.submit(self.__compute_noisy_sentences, dataset[i_start:i_end], i_start,
                                                   pbar, src_lang, tgt_lang))

                    # Update indexes
                    i_start = i_end
                    i_end = i_end + sentence_pairs_per_thread
                    if reminder > 0:
                        i_end = i_end + 1
                        reminder = reminder - 1

                # Concatenate all the noisy sentences' indexes that were found
                full_stop = []
                too_short = []
                empty = []
                too_different = []
                wrong_lang = []
                for result in futures:
                    result_dict = result.result()
                    full_stop.extend(result_dict["full_stop"])
                    too_short.extend(result_dict["too_short"])
                    empty.extend(result_dict["empty"])
                    too_different.extend(result_dict["too_different"])
                    wrong_lang.extend(result_dict["wrong_lang"])

        # Write the noisy sentences in their respective files
        self.__write_noisy_sentences_on_file(f"{dataset_name}_full_stop_sentences.txt", full_stop,
                                             dataset, src_lang, tgt_lang)
        self.__write_noisy_sentences_on_file(f"{dataset_name}_too_short_sentences.txt", too_short,
                                             dataset, src_lang, tgt_lang)
        self.__write_noisy_sentences_on_file(f"{dataset_name}_empty_sentences.txt", empty, dataset, src_lang, tgt_lang)
        self.__write_noisy_sentences_on_file(f"{dataset_name}_too_different_sentences.txt", too_different,
                                             dataset, src_lang, tgt_lang)
        self.__write_noisy_sentences_on_file(f"{dataset_name}_wrong_lang_sentences.txt", wrong_lang,
                                             dataset, src_lang, tgt_lang)
