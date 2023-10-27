import yaml
import os
import multiprocessing
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, Future
from tqdm.auto import tqdm
from typing import Dict, List


def _compute_collisions(
    dataset: Dict[str, List[Dict[str, str]]],
    lang: str,
    idx_start: int,
    pbar_tqdm: tqdm,
) -> List[int]:
    collision_indexes: List[int] = []
    sentence_pairs = dataset["translation"]
    for i, translation in enumerate(sentence_pairs):
        sent = translation[lang]
        if hash(sent) in string_hashes and sent == string_hashes[hash(sent)]:
            collision_indexes.append(idx_start + i)

        pbar_tqdm.update(1)

    return collision_indexes


if __name__ == "__main__":
    # Retrieve configurations
    path = os.path.dirname(__file__)
    with open(f"{path}/dedup.yaml") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    dataset_to_clean_dict = config["dataset_to_clean"]
    dataset_to_check_dict = config["dataset_to_check"]
    num_threads = config["num_threads"]
    lang_to_check = dataset_to_clean_dict["name"].split("-")[0]

    # Set the correct number of threads
    cpu_max_threads = multiprocessing.cpu_count()
    if num_threads > cpu_max_threads:
        num_threads = cpu_max_threads

    # Load datasets
    dataset_to_clean = load_dataset(**dataset_to_clean_dict)
    dataset_to_check = load_dataset(**dataset_to_check_dict)

    # Create hashes for those sentence to check for duplicates
    string_hashes: Dict[int, str] = dict()
    for sentence_pair in dataset_to_check:
        sentence: str = sentence_pair["translation"][lang_to_check]
        string_hashes[hash(sentence)] = sentence

    # Define number of sentence pairs per thread and start and end indexes
    sentence_pairs_per_thread, reminder = divmod(len(dataset_to_clean), num_threads)
    i_start, i_end = 0, sentence_pairs_per_thread

    # Check for duplicates in parallel
    futures: List[Future] = []
    with tqdm(total=len(dataset_to_clean)) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for thr in range(num_threads):
                futures.append(
                    executor.submit(_compute_collisions, dataset_to_clean[i_start:i_end], lang_to_check, i_start, pbar)
                )

                # Update indexes
                i_start = i_end
                i_end = i_end + sentence_pairs_per_thread
                if reminder > 0:
                    i_end = i_end + 1
                    reminder = reminder - 1

            # Concatenate all the duplicated sentences' indexes that were found
            duplicates_idxs = []
            for result in futures:
                duplicates_idxs.extend(result.result())

    print(f"Number of duplicates: {len(duplicates_idxs)}\nIndexes: {duplicates_idxs}")
