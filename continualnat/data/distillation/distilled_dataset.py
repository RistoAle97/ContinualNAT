import os

import datasets
from datasets.config import HF_DATASETS_CACHE


class DistilledDatasetConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        src_lang, tgt_lang = kwargs["name"].split("-")
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang


class DistilledDataset(datasets.GeneratorBasedBuilder):

    def __init__(self, src_lang: str, tgt_lang: str, path_name: str, *args, **kwargs):
        self.path_name = path_name
        self.BUILDER_CONFIGS = [
            DistilledDatasetConfig(
                name=f"{src_lang}-{tgt_lang}",
                description=f"Translating from {src_lang} to {tgt_lang} or viceversa",
                version=datasets.Version("1.0.0", ""),
            )
        ]
        if "max_train_samples" in kwargs and kwargs.get("cache_dir", None) is None:
            kwargs["cache_dir"] = os.path.join(
                str(HF_DATASETS_CACHE),
                f"trainsamples_{kwargs['max_train_samples']}",
            )

        self.max_samples = {"train": kwargs.get("max_train_samples", 2 ** 64)}
        kwargs = {k: v for k, v in kwargs.items() if k not in ["max_train_samples", "id_filter"]}
        super().__init__(*args, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description="Distilled dataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "translation": datasets.Translation(languages=[self.config.src_lang, self.config.tgt_lang])
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapath": self.path_name, "max_samples": self.max_samples["train"]}
            )
        ]

    def _generate_examples(self, datapath, max_samples):
        src_path = f"{datapath}.{self.config.src_lang}"
        tgt_path = f"{datapath}.{self.config.tgt_lang}"
        with open(src_path, encoding="utf-8") as f1, open(tgt_path, encoding="utf-8") as f2:
            for sentence_counter, (src_sentence, tgt_sentence) in enumerate(zip(f1, f2)):
                if sentence_counter == max_samples:
                    return

                sample = (
                    sentence_counter,
                    {
                        "id": sentence_counter,
                        "translation": {
                            self.config.src_lang: src_sentence.strip(),
                            self.config.tgt_lang: tgt_sentence.strip()}
                    }
                )
                yield sample
