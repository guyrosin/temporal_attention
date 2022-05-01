from typing import TYPE_CHECKING

from transformers.file_utils import _LazyModule

_import_structure = {
    "configuration_tempobert": [
        "TempoBertConfig",
    ],
}

_import_structure["tokenization_tempobert_fast"] = ["TempoBertTokenizerFast"]

_import_structure["modeling_tempobert"] = [
    "TempoBertForMaskedLM",
    "TempoBertModel",
    "TempoBertForPreTraining",
    "TempoBertForSequenceClassification",
    "TempoBertForTokenClassification",
]

if TYPE_CHECKING:
    from .configuration_tempobert import TempoBertConfig
    from .modeling_tempobert import (
        TempoBertForMaskedLM,
        TempoBertForPreTraining,
        TempoBertForSequenceClassification,
        TempoBertForTokenClassification,
        TempoBertModel,
    )
    from .tokenization_tempobert_fast import TempoBertTokenizerFast

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure
    )
