import numpy as np
import torch
from tqdm.std import trange

import hf_utils
import utils
from train_tempobert import ModelArguments
from transformers import AutoModelForMaskedLM

cache_dir = "data/cache"


class BertModel:
    def __init__(self, model_name_or_path=None, hf_pipeline=None, device=None) -> None:
        if hf_pipeline:
            self.model = hf_pipeline.model
            self.tokenizer = hf_pipeline.tokenizer
            self.device = self.model.device
            self.pipeline = hf_pipeline
        else:
            model_args = ModelArguments(
                model_name_or_path=model_name_or_path, cache_dir=cache_dir
            )
            self.model, self.tokenizer = hf_utils.load_pretrained_model(
                model_args, AutoModelForMaskedLM, expect_times_in_model=False
            )
            if device is None:
                # Use GPU if available
                device = 0 if torch.cuda.is_available() else -1
            self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)

        self.config = self.model.config

    def __str__(self):
        """Return a short version of the model's name or path"""
        return hf_utils.get_model_name(self.model.name_or_path)

    def encode_sentences(self, input, time=None, batch_size=None, return_batch=False):
        """Returns embedding(s) for the given input

        Args:
            input: A single text or a list of texts.
            time: A time point (str) or a list of time points.

        Returns:
            A tensor of embedding(s).
        """
        # this returns logits instead of hidden states (possibly a bug in Transformers), so I'm using my own code below.
        # result = self.extract_pipeline(input, time_id=time)
        kwargs = {}
        range_loop = (
            trange(0, len(input), batch_size)
            if batch_size and len(input) / batch_size > 5
            else range(1)
        )
        for i in range_loop:
            batch = input[i : i + batch_size] if batch_size else input
            if isinstance(time, str):
                batch_time = len(batch) * [time]
            elif isinstance(time, list):
                batch_time = time[i : i + batch_size] if batch_size else time
            else:
                batch_time = None
            if batch_time is not None:
                kwargs["time_id"] = batch_time
            batch = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                # add_special_tokens=False, # uncomment to get only tokens, without [CLS] and [SEP]
                return_tensors="pt",
                **kwargs,
            )
            batch = batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(**batch, output_hidden_states=True)
                if return_batch:
                    yield batch, model_output
                else:
                    yield model_output
            if batch_size:
                i += batch_size
            else:
                break

    def embed_word(
        self, input, word, time=None, batch_size=None, hidden_layers_number=1
    ):
        """Returns embedding(s) for the given word in the given input(s).

        Args:
            input: A single text or a list of texts.
            word: A single word to embed.
            time: A time point (str) or a list of time points.

        Returns:
            A tensor of embedding(s).
        """
        if not hasattr(self.model.config, 'times'):
            time = None  # ignore the given time if the model is not temporal
        batch_and_outputs = self.encode_sentences(
            input, time, batch_size, return_batch=True
        )
        result = torch.as_tensor([], device=self.device)
        # Note: remember that `result` will get bigger during the loop and will eventually contain the whole dataset,
        # so in case of an OOM exception, I can put it in the CPU. (I checked, it's faster in the GPU)
        if word in self.tokenizer.vocab:
            word_vocab_index = self.tokenizer.vocab[word]
            for batch, model_output in batch_and_outputs:
                last_hidden_states = torch.sum(
                    torch.stack(model_output.hidden_states[-hidden_layers_number:]), 0
                )
                # Extract the token embedding for the target word
                # Find the index of the target word in each sentence
                all_indices = (batch.data["input_ids"] == word_vocab_index).nonzero(
                    as_tuple=False
                )
                # `all_indices` is a matrix where each row is [row_index, word_index]
                sentence_indices = all_indices[:, 0]
                indices = all_indices[:, 1]
                # Select the embedding of the target word, in each sentence
                vecs = last_hidden_states[sentence_indices, indices, :]
                # vecs' shape is: (number of appearances of the word,  emb_dim)
                # assert vecs.shape == (batch_size, model.config.hidden_size)
                if vecs.shape[0] == 1:  # in case `input` is a single sentence
                    vecs = torch.squeeze(vecs)
                result = torch.cat((result, vecs))
        else:
            subword_vocab_indices = self.tokenizer.encode(
                word, add_special_tokens=False
            )
            for batch, model_output in batch_and_outputs:
                last_hidden_states = torch.sum(
                    torch.stack(model_output.hidden_states[-hidden_layers_number:]), 0
                )
                input_ids_all = batch.data["input_ids"].cpu().numpy()
                sent_to_tokens = {
                    sent_i: utils.search_sequence_numpy(
                        input_ids,
                        np.array(subword_vocab_indices),
                    )
                    for sent_i, input_ids in enumerate(input_ids_all)
                }
                all_indices = [
                    [sent, token]
                    for sent, tokens in sent_to_tokens.items()
                    for token in tokens
                ]
                all_indices = torch.as_tensor(all_indices, device=self.device)

                sentence_indices = all_indices[:, 0]
                indices = all_indices[:, 1]
                # Select the embedding of the target word, in each sentence
                vecs = last_hidden_states[sentence_indices, indices, :]
                # Take the average of the tokens of each word appearance
                n = len(subword_vocab_indices)
                assert vecs.shape[0] % n == 0
                # add a dimension for `n`, then take the mean by it
                # ref: https://stackoverflow.com/questions/15956309/averaging-over-every-n-elements-of-a-numpy-array
                vecs = torch.mean(
                    vecs.reshape(vecs.shape[0] // n, vecs.shape[1], n), axis=-1
                )
                # vecs' shape is: (number of appearances of the word,  emb_dim)
                # assert vecs.shape == (batch_size, model.config.hidden_size)
                if vecs.shape[0] == 1:  # in case `input` is a single sentence
                    vecs = torch.squeeze(vecs)
                result = torch.cat((result, vecs))
        return result
