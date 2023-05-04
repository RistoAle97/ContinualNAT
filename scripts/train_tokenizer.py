from datasets import load_dataset, concatenate_datasets, interleave_datasets
from tokenizers.implementations import SentencePieceBPETokenizer
from tokenizers.processors import ByteLevel


def batch_iterator(batch_size):
    batch = []
    for example in dataset:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []

    # yield last batch
    if batch:
        yield batch


if __name__ == "__main__":
    # Load datasets and concatenate them
    dataset_en = load_dataset("cc100", lang="en", split="train", verification_mode="no_checks", streaming=True)
    dataset_de = load_dataset("cc100", lang="de", split="train", verification_mode="no_checks", streaming=True)
    dataset_fr = load_dataset("cc100", lang="fr", split="train", verification_mode="no_checks", streaming=True)
    dataset_es = load_dataset("cc100", lang="es", split="train", verification_mode="no_checks", streaming=True)
    num_samples = 10000000
    dataset_en = dataset_en.take(num_samples)
    dataset_de = dataset_de.take(num_samples)
    dataset_fr = dataset_fr.take(num_samples)
    dataset_es = dataset_es.take(num_samples)
    # dataset = concatenate_datasets([dataset_en, dataset_de, dataset_fr, dataset_es])
    dataset = interleave_datasets([dataset_en, dataset_de, dataset_fr, dataset_es], stopping_strategy="all_exhausted")

    # Train the SentencepieceBPE tokenizer
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<length>", "<mask>"]
    vocab_size = 32000
    sentencepiece_tokenizer = SentencePieceBPETokenizer()
    sentencepiece_tokenizer.post_processor = ByteLevel()
    sentencepiece_tokenizer.train_from_iterator(batch_iterator(1000), vocab_size, special_tokens=special_tokens)
    sentencepiece_tokenizer.save(f"sentencepiece_config_{int(vocab_size / 1000)}k.json")
