import dataclasses
import importlib
import logging
import os
import sys
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import datasets
from loguru import logger

import utils
from tokenization_utils_base import LARGE_INTEGER
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    TOKENIZER_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import tokenization_auto
from transformers.models.auto.auto_factory import (
    _LazyAutoMapping,
    getattribute_from_module,
)
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    _LazyConfigMapping,
    model_type_to_module_name,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint, is_main_process, set_seed
from transformers.utils import logging as hf_logging


def set_transformers_logging(training_args=None, file_handler_path=None):
    """Set the verbosity to info of the Transformers logger."""
    if training_args:
        log_level = training_args.get_process_log_level()
        datasets.utils.logging.set_verbosity(log_level)
        hf_logging.set_verbosity(log_level)
    else:
        hf_logging.set_verbosity_info()
        datasets.utils.logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    if file_handler_path:
        file_handler = logging.FileHandler(file_handler_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        hf_logging.get_logger().addHandler(file_handler)


def _load_auto_config(model_args, data_args=None, num_labels=None, **kwargs):
    additional_kwargs = (
        dict(
            finetuning_task=dataclasses.asdict(data_args).get("task_name"),
            num_labels=num_labels,
        )
        if num_labels
        else {}
    )
    kwargs.update(additional_kwargs)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, **kwargs
    )
    return config


def load_pretrained_model(
    model_args,
    model_cls=AutoModelForMaskedLM,
    data_args=None,
    num_labels=None,
    times=None,
    expect_times_in_model=False,
    return_config=False,
    never_split=None,
    verbose=False,
    **kwargs,
):
    """Load a pretrained transformers model.

    Args:
        model_args (ModelArguments): A ModelArguments instance.
        model_cls (class): The class of the model to load.
        data_args (DataTrainingArguments, optional): A DataTrainingArguments instance.
        num_labels (int, optional): Number of labels.
        times (optional): List of time points.
        expect_times_in_model (bool, optional): True to require the model to contain its supported time points.
        return_config (bool, optional): True to return the config.
        kwargs: Additional arguments for the configuration.

    """
    if not verbose:
        current_log_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_warning()
    config = _load_auto_config(model_args, data_args, num_labels, **kwargs)
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if expect_times_in_model and not hasattr(config, 'times'):
        raise ValueError('The given model does not contain any time points')
    if hasattr(config, 'times'):
        times = config.times
    tokenizer_kwargs = dict(times=times) if times else {}
    if hasattr(config, 'time_embedding_type'):
        tokenizer_kwargs["time_embedding_type"] = config.time_embedding_type
    if never_split:
        tokenizer_kwargs["do_basic_tokenize"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        never_split=never_split,
        **tokenizer_kwargs,
    )
    if tokenizer.model_max_length > LARGE_INTEGER:
        # Hardcode model_max_length in case it wasn't saved with the pretrained model (happened for 'prajjwal1/bert-tiny')
        tokenizer.model_max_length = 128 if hasattr(config, 'times') else 512
    if verbose:
        logger.info(f"Loaded a pretrained model from {model_args.model_name_or_path}")
    else:
        hf_logging.set_verbosity(current_log_level)
    return (model, tokenizer, config) if return_config else (model, tokenizer)


def init_tempo_config(config, times, time_embedding_type):
    config.times = times
    config.time_embedding_type = time_embedding_type


def config_to_temporal(config, cls, times, time_embedding_type):
    config.__class__ = cls
    init_tempo_config(config, times, time_embedding_type)
    return config


class TempoModelName(Enum):
    TempoBert = auto()

    @classmethod
    def has_key(cls, name):
        return any(x for x in cls if x.name == name or x.name.lower() == name)


def lazyconfig_getitem(mapping, key):
    """
    Adapted from _LazyConfigMapping.__getitem__() in configuration_auto.py
    """
    if key not in mapping._mapping:
        raise KeyError(key)
    value = mapping._mapping[key]
    module_name = model_type_to_module_name(key)
    if module_name not in mapping._modules:
        if TempoModelName.has_key(key):
            mapping._modules[module_name] = importlib.import_module(
                f".{module_name}", "models"
            )
        else:
            mapping._modules[module_name] = importlib.import_module(
                f".{module_name}", "transformers.models"
            )
    return getattr(mapping._modules[module_name], value)


def _load_attr_from_module(mapping, model_type, attr):
    """
    Adapted from _LazyAutoMapping._load_attr_from_module() in auto_factory.py
    """
    module_name = model_type_to_module_name(model_type)
    if module_name not in mapping._modules:
        if TempoModelName.has_key(model_type):
            mapping._modules[module_name] = importlib.import_module(
                f".{module_name}", "models"
            )
        else:
            mapping._modules[module_name] = importlib.import_module(
                f".{module_name}", "transformers.models"
            )
    return getattribute_from_module(mapping._modules[module_name], attr)


def tokenizer_class_from_name(class_name: str):
    """
    Adapted from tokenizer_class_from_name() in auto_factory.py
    """
    if class_name == "PreTrainedTokenizerFast":
        return PreTrainedTokenizerBase

    for module_name, tokenizers in TOKENIZER_MAPPING_NAMES.items():
        if class_name in tokenizers:
            module_name = model_type_to_module_name(module_name)

            if TempoModelName.has_key(module_name):
                module = importlib.import_module(f".{module_name}", "models")
            else:
                module = importlib.import_module(
                    f".{module_name}", "transformers.models"
                )
            return getattr(module, class_name)

    return None


def register_temporal_classes():
    """Register the temporal classes to enable auto loading."""
    # Register configs
    CONFIG_MAPPING_NAMES[TempoModelName.TempoBert.name.lower()] = "TempoBertConfig"

    # Register tokenizers
    TOKENIZER_MAPPING_NAMES[TempoModelName.TempoBert.name.lower()] = (
        "BertTokenizer",
        "TempoBertTokenizerFast",
    )
    TOKENIZER_MAPPING._config_mapping[
        TempoModelName.TempoBert.name.lower()
    ] = "TempoBertConfig"
    TOKENIZER_MAPPING._reverse_config_mapping[
        "TempoBertConfig"
    ] = TempoModelName.TempoBert.name.lower()

    # Register models
    MODEL_MAPPING_NAMES[TempoModelName.TempoBert.name.lower()] = "TempoBertModel"
    MODEL_FOR_MASKED_LM_MAPPING_NAMES[
        TempoModelName.TempoBert.name.lower()
    ] = "TempoBertForMaskedLM"
    MODEL_FOR_MASKED_LM_MAPPING._config_mapping[
        TempoModelName.TempoBert.name.lower()
    ] = "TempoBertConfig"
    MODEL_FOR_MASKED_LM_MAPPING._reverse_config_mapping[
        "TempoBertConfig"
    ] = TempoModelName.TempoBert.name.lower()
    MODEL_FOR_MASKED_LM_MAPPING._model_mapping[
        TempoModelName.TempoBert.name.lower()
    ] = "TempoBertForMaskedLM"

    setattr(_LazyConfigMapping, '__getitem__', lazyconfig_getitem)
    setattr(_LazyAutoMapping, '_load_attr_from_module', _load_attr_from_module)
    setattr(tokenization_auto, 'tokenizer_class_from_name', tokenizer_class_from_name)


setattr(PretrainedConfig, 'init_tempo_config', init_tempo_config)


def init_run(
    parser,
    date_str=None,
    args=None,
    args_filename=None,
):
    """Load arguments, initialize logging and set seed."""
    (model_args, data_args, training_args,) = parser.parse_args_into_dataclasses(
        look_for_args_file=True,
        args_filename=args_filename,
    )

    # Set fp16 to True if cuda is available
    if training_args.device.type == "cuda":
        training_args.fp16 = True
    # Set dataloader_pin_memory to False if no_cuda is set (o.w. an exception will be thrown in the DataLoader)
    if training_args.no_cuda:
        training_args.dataloader_pin_memory = False
    # special_tokens_mask was getting removed from the dataset and it hurt performance -> don't remove unused columns
    training_args.remove_unused_columns = False

    for args_instance in (model_args, data_args, training_args):
        if args is not None:
            # update the argument dataclasses according to the given arguments.
            keys = {f.name for f in dataclasses.fields(args_instance)}
            for key, val in args.items():
                if key in keys:
                    args_instance.__dict__[key] = val

    if not date_str:
        date_raw = datetime.now()
        date_str = f"{date_raw.year}-{date_raw.month}-{date_raw.day}_{date_raw.hour}-{date_raw.minute}-{date_raw.second}"

    training_args.output_dir = str(Path(training_args.output_dir) / date_str)

    if is_main_process(training_args.local_rank):
        logger.add(f"{training_args.logging_dir}/training_log_{date_str}.log")
    else:
        # Show only warnings (at least)
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    utils.set_result_logger_level()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}"
    )
    for args_instance in (model_args, data_args, training_args):
        logger.info(args_instance)

    # Set the Transformers logger (on main process only)
    file_handler_path = f"{training_args.logging_dir}/hf_log_{date_str}.log"
    set_transformers_logging(training_args, file_handler_path=file_handler_path)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model
    set_seed(training_args.seed)

    register_temporal_classes()

    return model_args, data_args, training_args, last_checkpoint


def get_model_name(model_name_or_path):
    """Return a short version of the model's name or path"""
    path = Path(model_name_or_path)
    if path.exists():
        if "checkpoint-" in path.name:
            model_name_or_path = f"{path.parent.name}/{path.name}"
        else:
            model_name_or_path = str(path.name)
    return model_name_or_path


def prepare_tf_classes():
    set_transformers_logging()
    register_temporal_classes()
