#!/usr/bin/env python
"""
Training script for temporal BERT model using temporal attention.

Based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
"""
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import datasets
from dotenv import load_dotenv
from loguru import logger

import data_utils
import hf_utils
from models.tempobert.modeling_tempobert import TempoBertForMaskedLM
from temporal_data_collator import DataCollatorForTimePrependedLanguageModeling
from temporal_text_dataset import TemporalText
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    freeze_layers: Optional[str] = field(
        default=False,
        metadata={
            "help": "True to freeze all encoder layers, or a string specifying the layer numbers to freeze."
        },
    )
    hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "Dimensionality of the encoder layers and the pooler layer."},
    )
    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={"help": "Number of hidden layers in the Transformer encoder."},
    )
    tokenizer: Optional[str] = field(
        default='bert-base',
        metadata={
            "help": "Tokenizer name without case, e.g., `bert-base`. Use `cased_tokenizer` to specify the case."
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    time_embedding_type: Optional[str] = field(
        default="temporal_attention",
        metadata={
            "help": "Time embedding type. Possible values: `prepend_token`, `temporal_attention`."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="temporal_text_dataset.py",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or directory."}
    )
    validation_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file or directory to evaluate the perplexity on."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    time_mlm_probability: Optional[float] = field(
        default=None,
        metadata={
            "help": "Ratio of time tokens to mask for masked language modeling loss (relevant in case of a time-prepended model). "
            "If None, time tokens are occasionally masked, like any other token."
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    times: Optional[str] = field(
        default=None, metadata={"help": "List of time points for the model to use."}
    )
    words_for_vocab_file: Optional[str] = field(
        default=None,
        metadata={"help": "Text file containing words to add to the model vocabulary."},
    )
    corpus_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the corpus (e.g., liverpool, semeval_eng)."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_path is None
            and self.validation_path is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation path."
            )


def freeze_model_layers(model, freeze_layers_arg):
    if freeze_layers_arg:
        if isinstance(freeze_layers_arg, bool):
            for layer in model.base_model.encoder.layer:
                for param in layer.parameters():
                    param.requires_grad = False
        elif isinstance(freeze_layers_arg, str):
            layer_indexes = [int(x) for x in freeze_layers_arg.split(",")]
            for layer_idx in layer_indexes:
                for param in list(
                    model.base_model.encoder.layer[layer_idx].parameters()
                ):
                    param.requires_grad = False


def tokenize_dataset_line_by_line(
    dataset,
    data_args,
    training_args,
    tokenizer,
    text_column_name,
    column_names,
    max_seq_length,
    return_special_tokens_mask,
):
    """Tokenize each nonempty line."""

    def _tokenize(examples, data_args, tokenizer, text_column_name):
        padding = "max_length" if data_args.pad_to_max_length else False
        return tokenizer(
            examples[text_column_name],
            examples['time'],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=return_special_tokens_mask,
        )

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if line and not line.isspace()
        ]
        return _tokenize(examples, data_args, tokenizer, text_column_name)

    with training_args.main_process_first(desc="dataset map tokenization"):
        return dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset line_by_line",
        )


def tokenize_dataset_concat(
    dataset,
    data_args,
    training_args,
    tokenizer,
    text_column_name,
    column_names,
    max_seq_length,
    return_special_tokens_mask,
):
    """Tokenize every text, then concatenate them together before splitting them in smaller parts."""

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            examples['time'],
            return_special_tokens_mask=return_special_tokens_mask,
        )

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

    # Concatenate all texts from our dataset and generate chunks of max_seq_length
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len
        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    with training_args.main_process_first(desc="grouping texts together"):
        return tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )


def load_data(
    corpus_path,
    data_args,
    training_args,
    model_args,
    tokenizer,
):
    dataset_files = data_utils.iterdir(corpus_path, suffix=".txt", to_str=True)
    logger.info("Loading dataset files...")
    dataset = datasets.load_dataset(
        data_args.dataset_name,
        data_files=dataset_files,
        split="train",  # Note the split is always labeled "train"
        cache_dir=model_args.cache_dir,
    )
    logger.info(f"Loaded dataset of {dataset.num_rows:,} rows. Preprocessing...")
    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # DataCollatorForLanguageModeling is more efficient when it receives the `special_tokens_mask`.
    return_special_tokens_mask = True
    if data_args.line_by_line:
        tokenized_dataset = tokenize_dataset_line_by_line(
            dataset,
            data_args,
            training_args,
            tokenizer,
            text_column_name,
            column_names,
            max_seq_length,
            return_special_tokens_mask,
        )
    else:
        tokenized_dataset = tokenize_dataset_concat(
            dataset,
            data_args,
            training_args,
            tokenizer,
            text_column_name,
            column_names,
            max_seq_length,
            return_special_tokens_mask,
        )
    return tokenized_dataset


def train_tempobert():
    """Main training function for temporal BERT"""

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args, last_checkpoint = hf_utils.init_run(parser)

    if (training_args.do_eval and not data_args.validation_path) or (
        not training_args.do_eval and data_args.validation_path
    ):
        logger.error(f"{training_args.do_eval=} but {data_args.validation_path=}")
        exit()

    dataset_files = data_utils.iterdir(data_args.train_path, suffix=".txt")

    if data_args.times:
        if ',' in data_args.times:
            times = data_args.times.split(',')
        elif '-' in data_args.times:
            from_time, to_time = data_args.times.split('-')
            times = list(map(str, range(from_time, to_time + 1)))
        else:
            times = [data_args.times]
    else:
        times = sorted([TemporalText.find_time(f.name) for f in dataset_files])
    logger.info(f'Loaded {len(times)} time points from {data_args.train_path}.')

    # Set the model and data collator classes
    pad_to_multiple_of_8 = (
        data_args.line_by_line
        and training_args.fp16
        and not data_args.pad_to_max_length
    )
    data_collator_cls = (
        DataCollatorForTimePrependedLanguageModeling
        if model_args.time_embedding_type.startswith("prepend_")
        else DataCollatorForLanguageModeling
    )

    # Load the config, model, and tokenizer.
    logger.info(f"Training from a pretrained {model_args.model_name_or_path} model")
    model, tokenizer, config = hf_utils.load_pretrained_model(
        model_args, data_args=data_args, return_config=True
    )
    # Convert all components to temporal
    temporal_model_type = (
        config.model_type
        if config.model_type.startswith("tempo")
        else f"tempo{config.model_type}"
    )
    temporal_config_class = CONFIG_MAPPING[temporal_model_type]
    temporal_tokenizer_fast_class = TOKENIZER_MAPPING[temporal_config_class][1]
    config = hf_utils.config_to_temporal(
        config,
        temporal_config_class,
        times=times,
        time_embedding_type=model_args.time_embedding_type,
    )
    tokenizer = temporal_tokenizer_fast_class.from_non_temporal(tokenizer, config)
    model = TempoBertForMaskedLM.from_non_temporal(model, config)

    if data_args.words_for_vocab_file:
        tokens = Path(data_args.words_for_vocab_file).read_text().splitlines()
        if tokenizer.do_lower_case:
            tokens = [t.lower() for t in tokens]
        num_added_toks = tokenizer.add_tokens(tokens)
        logger.info(
            f"Added {num_added_toks} tokens from {data_args.words_for_vocab_file} to the vocabulary"
        )

    # Necessary only if new words were introduced by the tokenizer
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_data(
        data_args.train_path,
        data_args,
        training_args,
        model_args,
        tokenizer,
    )
    eval_dataset = (
        load_data(
            data_args.validation_path,
            data_args,
            training_args,
            model_args,
            tokenizer,
        )
        if training_args.do_eval and data_args.validation_path
        else None
    )

    # The data collator takes care of randomly masking tokens
    kwargs = {}
    if data_collator_cls == DataCollatorForTimePrependedLanguageModeling:
        kwargs["different_time_mlm"] = data_args.time_mlm_probability is not None
        kwargs["time_mlm_probability"] = data_args.time_mlm_probability
        kwargs["time_tokens"] = [f"<{time}>" for time in times]
    data_collator = data_collator_cls(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        **kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info(f"Training... Output folder: {training_args.output_dir}")
        start = datetime.now()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info(f"Done training! Elapsed time: {datetime.now() - start}")
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    load_dotenv()
    train_tempobert()
