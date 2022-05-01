from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch

from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    _torch_collate_batch,
)
from transformers.tokenization_utils_base import BatchEncoding


@dataclass
class DataCollatorForTimePrependedLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling where a time token is prepended to the input.
    Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.

    Args:
        different_time_mlm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use a different bank for masking the time tokens. If set to :obj:`False`, this class behaves the same as `DataCollatorForLanguageModeling`.
        time_mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask time tokens in the input, when :obj:`different_time_mlm` is set to :obj:`True`.
    """

    different_time_mlm: bool = False
    time_mlm_probability: float = 0.15
    time_tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.times_tensor = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(self.time_tokens)
        )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if not self.different_time_mlm:
            return super().__call__(examples)

        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            # Remove the time tokens (the 2nd column)
            input_ids_without_time = torch.cat(
                (batch["input_ids"][:, :1], batch["input_ids"][:, 2:]), dim=1
            )
            special_mask_without_time = torch.cat(
                (special_tokens_mask[:, :1], special_tokens_mask[:, 2:]), dim=1
            )
            # Apply masking on the inputs and on the time tokens, separately
            time_tokens = batch["input_ids"][:, 1]
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                input_ids_without_time, special_tokens_mask=special_mask_without_time
            )
            new_time_tokens, time_labels = self.mask_time_tokens(time_tokens)
            # Insert the new time tokens to the batch
            batch["input_ids"] = torch.cat(
                (
                    batch["input_ids"][:, :1],
                    new_time_tokens.reshape((len(new_time_tokens), 1)),
                    batch["input_ids"][:, 1:],
                ),
                dim=1,
            )
            batch["labels"] = torch.cat(
                (
                    batch["labels"][:, :1],
                    time_labels.reshape((len(time_labels), 1)),
                    batch["labels"][:, 1:],
                ),
                dim=1,
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_time_tokens(
        self, times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens times/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = times.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.time_mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked time tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        times[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked time tokens with random time
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.times_tensor), labels.shape, dtype=torch.long
        )
        times[indices_random] = self.times_tensor[random_words[indices_random]]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return times, labels
