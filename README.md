# Temporal Attention for Language Models

This repository provides a reference implementation of the paper:

> Temporal Attention for Language Models<br>
Guy D. Rosin and and Kira Radinsky<br>
Findings of NAACL 2022, to appear<br>
Preprint: https://arxiv.org/abs/2202.02093

Abstract:
>Pretrained language models based on the transformer architecture have shown great success in NLP. Textual training data often comes from the web and is thus tagged with time-specific information, but most language models ignore this information.
They are trained on the textual data alone, limiting their ability to generalize temporally.<br>
In this work, we extend the key component of the transformer architecture, i.e., the self-attention mechanism, and propose temporal attention - a time-aware self-attention mechanism.
Temporal attention can be applied to any transformer model and requires the input texts to be accompanied with their relevant time points.<br>
This mechanism allows the transformer to capture this temporal information and create time-specific contextualized word representations.
We leverage these representations for the task of semantic change detection; we apply our proposed mechanism to BERT and experiment on three datasets in different languages (English, German, and Latin) that also vary in time, size, and genre.
Our proposed model achieves state-of-the-art results on all the datasets.


## Prerequisites

- Create an Anaconda environment with Python 3.8 and install requirements: 

        conda create -n tempo_att python=3.8
        conda activate tempo_att
        conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
        pip install -r requirements.txt

- Obtain datasets for training and evaluation on semantic change detection: [SemEval-2020 Task 1 datasets](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/).

## Usage

- Train BERT with temporal attention using `train_tempobert.py`. This script is similar to Hugging Face's language modeling training script ([link](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py)), and introduces a `time_embedding_type` argument, that is set to `temporal_attention` by default.
- Evaluate the trained model on semantic change detection using `semantic_change_detection.py`.

## Pointers

- The temporal attention mechanism is implemented in `modeling_tempobert.py`, in the `TemporalSelfAttention` class.
- Note: this repository also supports the TempoBERT model ([paper](https://arxiv.org/abs/2110.06366)), which prepends time tokens to text sequences: https://github.com/guyrosin/tempobert
