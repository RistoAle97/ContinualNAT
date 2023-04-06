from datasets import load_dataset, concatenate_datasets
from tokenizers.implementations import SentencePieceBPETokenizer
from tokenizers.processors import ByteLevel


def batch_iterator(batch_size):
    batch = []
    for example in dataset:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:  # yield last batch
        yield batch


if __name__ == "__main__":
    # Load datasets and concatenate them
    dataset_en = load_dataset("cc100", lang="en", split="train",
                              cache_dir="/disk1/a.ristori/cc100", verification_mode="no_checks", streaming=True)
    dataset_de = load_dataset("cc100", lang="de", split="train",
                              cache_dir="/disk1/a.ristori/cc100", verification_mode="no_checks", streaming=True)
    dataset_fr = load_dataset("cc100", lang="fr", split="train",
                              cache_dir="/disk1/a.ristori/cc100", verification_mode="no_checks", streaming=True)
    dataset_es = load_dataset("cc100", lang="es", split="train",
                              cache_dir="/disk1/a.ristori/cc100", verification_mode="no_checks", streaming=True)
    num_samples = 1000000
    dataset_en = dataset_en.take(num_samples)
    dataset_de = dataset_de.take(num_samples)
    dataset_fr = dataset_fr.take(num_samples)
    dataset_es = dataset_es.take(num_samples)
    dataset = concatenate_datasets([dataset_en, dataset_de, dataset_fr, dataset_es])

    # Train the SentencepieceBPE tokenizer
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<length>", "<mask>"]
    vocab_size = 32000
    sentencepiece_tokenizer = SentencePieceBPETokenizer()
    sentencepiece_tokenizer.post_processor = ByteLevel()
    sentencepiece_tokenizer.train_from_iterator(batch_iterator(1000), vocab_size, special_tokens=special_tokens)
    sentencepiece_tokenizer.save(f"sentencepiece_config_{vocab_size / 1000}k.json")
