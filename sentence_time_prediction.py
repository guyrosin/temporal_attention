"""Sentence time prediction evaluation"""
import pprint
import re
from collections.abc import Iterator
from functools import partial
from pathlib import Path

from loguru import logger
from sklearn.metrics import classification_report
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import data_utils
import hf_utils
import test_bert
import utils


def predict_time(time_to_label, fill_mask_pipeline, time_pattern, sentence):
    result_dict = test_bert.predict_time(
        sentence, fill_mask_pipeline, print_results=False
    )
    tokens = list(result_dict.keys())
    # Choose the token with the highest probability
    pred_token = tokens[0]
    pred = time_to_label[time_pattern.search(pred_token).group(1) if pred_token else 0]
    return pred


def sentence_time_prediction(
    train_corpus_path,
    test_corpus_path,
    fill_mask_pipelines,
    test_size=None,
    train_size=None,
    corpus_name=None,
    parallel=False,
):
    """
    For each sentence: predict its time.
    Note: The data size arguments are irrelevant for this method; they're used just to load a preprocessed dataset.
    """
    _, test_dataset = data_utils.split_temporal_dataset_files(
        train_corpus_path,
        test_corpus_path,
        corpus_name,
        train_size,
        test_size,
    )
    time_list = sorted(test_dataset.keys())
    time_pattern = re.compile(r"<(.+)>")
    time_to_label = {time: i for i, time in enumerate(time_list)}
    if not isinstance(fill_mask_pipelines, Iterator):
        fill_mask_pipelines = [fill_mask_pipelines]
    model_to_ys = {}

    def get_results_str(model_name, y_true, y_pred):
        accuracy = utils.calc_accuracy(y_true, y_pred)
        f1_macro = utils.calc_f1(y_true, y_pred, average="macro")
        return f"{model_name} accuracy: {accuracy:.2%}, macro-f1: {f1_macro:.2%}"

    for fill_mask_pipeline in fill_mask_pipelines:
        model_name = hf_utils.get_model_name(fill_mask_pipeline.model.name_or_path)
        logger.info(f"Model: {model_name}")
        ys = []
        for time in time_list:
            sentences = list(test_dataset[time])
            y_true = [time_to_label[time]] * len(sentences)
            predict_time_partial = partial(
                predict_time, time_to_label, fill_mask_pipeline, time_pattern
            )
            if parallel:
                y_pred = process_map(predict_time_partial, sentences, max_workers=4)
            else:
                with tqdm(sentences, desc=time) as t:
                    y_pred = []
                    for i, sent in enumerate(t):
                        pred = predict_time_partial(sent)
                        y_pred.append(pred)
                        if i % 100 == 0 and i > 0:
                            utils.calc_accuracy(y_true, y_pred, tqdm_bar=t)
            utils.calc_accuracy(y_true, y_pred, tqdm_bar=t)
            ys.append((y_true, y_pred))
        y_true, y_pred = (
            sum((y_true for y_true, _ in ys), []),
            sum((y_pred for _, y_pred in ys), []),
        )
        logger.info(get_results_str(model_name, y_true, y_pred))
        model_to_ys[model_name] = (y_true, y_pred)

    logger.info("Final results:")
    for model_name, (y_true, y_pred) in model_to_ys.items():
        logger.info(f"{model_name}:")
        logger.info(classification_report(y_true, y_pred))
    for model_name, (y_true, y_pred) in model_to_ys.items():
        logger.info(get_results_str(model_name, y_true, y_pred))


if __name__ == "__main__":
    hf_utils.prepare_tf_classes()

    data_path = "data/semeval_eng"

    corpus_name = Path(data_path).name

    train_corpus_path = f"{data_path}/{corpus_name}_train"
    test_corpus_path = f"{data_path}/{corpus_name}_test"
    device = 0

    # Train and test sizes (specify None to use the whole sets)
    test_size = None
    train_size = None

    MODEL_PATH = ""  # Path to your model

    logger.info(f"Will evaluate: {pprint.pformat(MODEL_PATH)}")

    tester = test_bert.Tester(MODEL_PATH, device=device)
    sentence_time_prediction(
        train_corpus_path,
        test_corpus_path,
        tester.fill_mask_pipelines,
        test_size=test_size,
        train_size=train_size,
        corpus_name=corpus_name,
    )
