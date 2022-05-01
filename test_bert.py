from typing import Iterable

import hf_utils
from bert_model import BertModel
from train_tempobert import ModelArguments
from transformers import AutoModelForMaskedLM, pipeline


def predict_time(sentence, fill_mask_pipelines, print_results=True):
    if not isinstance(fill_mask_pipelines, list):
        fill_mask_pipelines = [fill_mask_pipelines]
    time_tokens = [f"<{time}>" for time in fill_mask_pipelines[0].model.config.times]
    result_dict = {}
    original_sentence = sentence
    sentence = "[MASK] " + sentence
    for model_i, fill_mask in enumerate(fill_mask_pipelines):
        fill_result = fill_mask(sentence, targets=time_tokens, truncation=True)
        result = {res["token_str"]: res["score"] for res in fill_result}
        if len(fill_mask_pipelines) == 1:
            result_dict = result
        else:
            result_dict[model_i] = result
        if print_results:
            res_str = ', '.join(
                f'{token} ({score:.2f})' for token, score in result.items()
            )
            if len(fill_mask_pipelines) > 1:
                print(f"{model_i}: {original_sentence}: {res_str}")
            else:
                print(f"{original_sentence}: {res_str}")
    return result_dict


def load_model(model_name_or_path, expect_times_in_model=True):
    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    config_kwargs = {}
    model, tokenizer = hf_utils.load_pretrained_model(
        model_args,
        AutoModelForMaskedLM,
        expect_times_in_model=expect_times_in_model,
        **config_kwargs,
    )
    return model, tokenizer


class Tester:
    def __init__(self, model, device=-1, preload=False) -> None:
        hf_utils.prepare_tf_classes()
        if not isinstance(model, list):
            model = [model]
        model_tokenizer_list = (
            load_model(m, expect_times_in_model=False) for m in model
        )
        if preload:
            model_tokenizer_list = list(model_tokenizer_list)
        self.fill_mask_pipelines = (
            pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
            for model, tokenizer in model_tokenizer_list
        )
        if preload:
            self.fill_mask_pipelines = list(self.fill_mask_pipelines)
        self.bert_models = (
            BertModel(hf_pipeline=fill_mask_pipeline, device=device)
            for fill_mask_pipeline in self.fill_mask_pipelines
        )
        if preload:
            self.bert_models = list(self.bert_models)
