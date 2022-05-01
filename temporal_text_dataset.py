import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

import datasets

FEATURES = datasets.Features(
    {
        "text": datasets.Value("string"),
        "time": datasets.Value("string"),
    }
)

ARROW_SCHEMA = pa.schema({"text": pa.string(), "time": pa.string()})


@dataclass
class TextConfig(datasets.BuilderConfig):
    """BuilderConfig for text files."""

    encoding: str = "utf-8"
    chunksize: int = 10 << 20  # 10MB
    keep_linebreaks: bool = False


class TemporalText(datasets.ArrowBasedBuilder):
    """Dataset for text with time points.

    Based on the Text dataset (https://github.com/huggingface/datasets/blob/master/src/datasets/packaged_modules/text/text.py).
    """

    BUILDER_CONFIG_CLASS = TextConfig

    def _info(self):
        return datasets.DatasetInfo(features=FEATURES)

    @staticmethod
    def find_time(filename):
        filename = Path(filename).name
        # Look for the longest string starting with a digit, until a dot or an alphabet character.
        # This matches both "nyt_2017.txt" and "nyt_2_2017.txt"
        m = re.match(r".*?_(\d+.*?)[\.a-zA-Z]", filename)
        if m is None:
            return None
        time = m.group(1)
        # Remove trailing underscores (e.g., for "nyt_2017_train.txt")
        time = time.strip("_")
        return time

    def _generate_tables(self, files):
        for file_idx, file in enumerate(files):
            batch_idx = 0
            with open(file, 'r', encoding=self.config.encoding) as f:
                time = self.find_time(f.name)
                while True:
                    batch = f.read(self.config.chunksize)
                    if not batch:
                        break
                    batch += f.readline()  # finish current line
                    batch = batch.splitlines()
                    pa_table = pa.Table.from_arrays(
                        [pa.array(batch), pa.array(len(batch) * [time])],
                        schema=ARROW_SCHEMA,
                    )
                    yield (file_idx, batch_idx), pa_table
                    batch_idx += 1

    def _split_generators(self, dl_manager):
        """The `data_files` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].
        If str or List[str], then the dataset returns only the 'train' split.
        If dict, then keys should be from the `datasets.Split` enum.
        """
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got data_files={self.config.data_files}"
            )
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs={"files": files}
                )
            ]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(
                datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files})
            )
        return splits

    def _generate_examples(self, **kwargs):
        raise NotImplementedError()
