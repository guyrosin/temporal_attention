import pickle
import random
from collections import defaultdict
from pathlib import Path

from datasets.dataset_dict import DatasetDict
from loguru import logger
from natsort import natsorted
from textsearch import TextSearch

import datasets
import utils
from temporal_text_dataset import TemporalText


def iterdir_one_folder(path, suffix=None, suffixes=None, to_str=False):
    """
    Given a path, return a list containing it, if it's a file, or all the files it contains, if it's a folder.
    The files are natrual sorted, and can be filtered by a given suffix(es).
    """
    path = Path(path)
    files = (
        [path]
        if path.is_file()
        else [
            x
            for x in natsorted(path.iterdir(), key=lambda x: x.stem)
            if (suffix and x.suffix == suffix)
            or (suffixes and x.suffix in suffixes)
            or (not suffix and not suffixes)
        ]
    )
    if to_str:
        files = [str(f) for f in files]
    return files


def iterdir(path, suffix=None, suffixes=None, to_str=False):
    """
    Given a path, return a list containing it, if it's a file, or all the files it contains, if it's a folder.
    The files are natrual sorted, and can be filtered by a given suffix(es).
    """
    if '*' in str(path):
        all_paths = list(Path('.').glob(path))
        all_paths = natsorted(all_paths, key=lambda x: x.stem)
    else:
        all_paths = [path]
    files = []
    for path in all_paths:
        files.extend(iterdir_one_folder(path, suffix, suffixes, to_str))
    return files


def load_temporal_dataset(
    path,
    size_per_time=None,
    dataset_to_exclude=None,
):
    dataset_files = iterdir(path, suffix=".txt")
    time_to_sentences = {}
    for file in dataset_files:
        time = TemporalText.find_time(file)
        prefix = f"{time=}: " if time else ""
        current_sentences = set(file.read_text().splitlines())
        if dataset_to_exclude:
            sentences_to_exclude = set(dataset_to_exclude[time])
            intersection = set.intersection(current_sentences, sentences_to_exclude)
            current_sentences -= intersection
            logger.info(
                f"{prefix}Excluded {len(intersection)} sentences that exist in the training set, out of {len(current_sentences)}"
            )
        if size_per_time:
            if len(current_sentences) > size_per_time:
                current_sentences = random.sample(current_sentences, size_per_time)
            elif len(current_sentences) < size_per_time:
                logger.warning(
                    f"{prefix}Not enough rows ({len(current_sentences)}), so skipping it."
                )
                continue
        time_to_sentences[time] = current_sentences
    logger.info(
        f"Loaded a dataset of times={list(time_to_sentences.keys())} with a total of {utils.count_dict_values(time_to_sentences):,} sentences"
    )
    return time_to_sentences


def get_dataset_path(train_path, corpus_name, train_size=None, test_size=None):
    path_name = f"{corpus_name}_split"
    if train_size:
        path_name += f"_{utils.millify(train_size)}"
    path_name += f"-{utils.millify(test_size)}" if test_size else "-all"
    return Path(train_path).parent / path_name


def load_train_test_datasets(train_path, test_path, cache_dir):
    files = iterdir(train_path, to_str=True)
    train_dataset = datasets.load_dataset(
        "temporal_text_dataset.py",
        data_files=files,
        split='train',
        cache_dir=cache_dir,
    )
    files = iterdir(test_path, to_str=True)
    test_dataset = datasets.load_dataset(
        "temporal_text_dataset.py",
        data_files=files,
        split='train',
        cache_dir=cache_dir,
    )
    dataset = DatasetDict({"train": train_dataset, "validation": test_dataset})
    return dataset


def split_temporal_dataset_files(
    train_path, test_path, corpus_name, train_size=None, test_size=None
):
    """
    Note: train and test sizes are per time point.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)
    dataset_path = get_dataset_path(train_path, corpus_name, train_size, test_size)
    exclude_similar_sentences = True if corpus_name.startswith("liverpool") else False
    out_train_path = dataset_path / train_path.name
    out_test_path = dataset_path / test_path.name
    if Path(out_train_path).exists() and Path(out_test_path).exists():
        datasets = [
            load_temporal_dataset(path) for path in (out_train_path, out_test_path)
        ]
        logger.info(f"Loaded preprocessed dataset from {dataset_path}")
    else:
        # Get the datasets
        logger.info("Loading dataset files...")
        datasets = []
        dataset_to_exclude = None
        for path, out_path, size in [
            (train_path, out_train_path, train_size),
            (test_path, out_test_path, test_size),
        ]:
            dataset = load_temporal_dataset(
                path,
                size_per_time=size,
                dataset_to_exclude=dataset_to_exclude,
                exclude_similar_sentences=exclude_similar_sentences,
            )
            out_path.mkdir(parents=True, exist_ok=True)
            for time, sentences in dataset.items():
                file_path = out_path / f"{corpus_name}_{time}.txt"
                file_path.write_text("\n".join(sentences))
            datasets.append(dataset)
            dataset_to_exclude = dataset
        logger.info(f"Saved split dataset to {dataset_path}")
    return datasets


def find_sentences_of_words_in_file(
    text_file, words, max_sentences, ignore_case=False, persist=True, override=False
):
    """Find sentences in a text file that contain the given words."""
    text_file = Path(text_file)
    case_str = "" if ignore_case else "_cased"
    file_path = text_file.with_name(f"{text_file.stem}_word_sentences{case_str}.pkl")
    if file_path.exists() and not override:
        with open(file_path, 'rb') as f:
            word_sentences = pickle.load(f)
    else:
        case = "ignore" if ignore_case else "sensitive"
        # Note: "ignore" means text matches will always be returned in lowercase.
        ts = TextSearch(case=case, returns="match")
        word_sentences = {word: [] for word in words}
        ts.add(words)
        with open(text_file) as fp:
            for sentence in fp:
                sentence = sentence.strip()
                found_words = ts.findall(sentence)
                for word in found_words:
                    word_sentences[word].append(sentence)
        for word, sentences in word_sentences.items():
            if len(sentences) > max_sentences:
                word_sentences[word] = random.sample(sentences, max_sentences)
        if persist:
            with open(file_path, 'wb') as f:
                pickle.dump(word_sentences, f)
            logger.debug(f"Sentences saved to {file_path}")
    return word_sentences


def find_sentences_of_words(
    text_files,
    words,
    max_sentences_per_time,
    ignore_case=False,
    override=False,
):
    """Find sentences in a given temporal corpus that contain the given words."""
    data_folder = Path(text_files[0]).parent
    case_str = "" if ignore_case else "_cased"
    file_path = (
        data_folder / f"word_time_{max_sentences_per_time}sentences{case_str}.pkl"
    )
    if Path(file_path).exists() and not override:
        with open(file_path, 'rb') as f:
            logger.debug(f"Loading word_time_sentences from {file_path}")
            word_time_sentences = pickle.load(f)
    else:
        logger.info(f"Finding relevant sentences in the corpus...")
        word_time_sentences = defaultdict(dict)
        for file in text_files:  # For each time period
            time = TemporalText.find_time(file)
            if not time:
                continue
            word_sentences = find_sentences_of_words_in_file(
                file,
                words,
                max_sentences_per_time,
                ignore_case=ignore_case,
                persist=False,
                override=False,
            )
            for word, sentences in word_sentences.items():
                word_time_sentences[word][time] = sentences
        with open(file_path, 'wb') as f:
            pickle.dump(word_time_sentences, f)
        logger.debug(f"Sentences saved to {file_path}")
    return word_time_sentences
