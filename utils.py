import math
import sys

import numpy as np
import typeguard
from loguru import logger


def set_result_logger_level():
    if "RESULT" not in logger._core.levels:
        logger.level("RESULT", no=60, color="<cyan>")


def millify(num, precision=0):
    """Humanize number (e.g., 30000 --> 30k).
    Taken from https://github.com/azaitsev/millify/tree/master/millify"""
    millnames = ['', 'k', 'M', 'B', 'T', 'P', 'E', 'Z', 'Y']
    num = float(num)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if num == 0 else math.log10(abs(num)) / 3)),
        ),
    )
    result = '{:.{precision}f}'.format(num / 10 ** (3 * millidx), precision=precision)
    return '{0}{dx}'.format(result, dx=millnames[millidx])


def check_type(value, expected_type, argname=None):
    """Ensure that value matches expected_type.
    Needed as Python currently doesn't have out of the box support for type checking using the typing module:
    https://stackoverflow.com/questions/55503673/how-do-i-check-if-a-value-matches-a-type-in-python

    Args:
        value: value to be checked against `expected_type`
        expected_type: a class or generic type instance
        argname: name of the argument to check; used for error messages

    Returns:
        True if the type is correct, otherwise False
    """
    if argname is None:
        argname = 'variable'
    try:
        typeguard.check_type(argname, value, expected_type)
        return True
    except:
        return False


def search_sequence_numpy(arr, seq):
    """Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.

    Taken from https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array/36535397#36535397
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return []  # No match found


def count_dict_values(d):
    return sum(len(v) for v in d.values())


def set_loguru_level(level="DEBUG"):
    logger.remove()
    logger.add(sys.stderr, level=level)


def get_correlation_str(scores, ground_truth, func, label, include_pvalue=False):
    r, p = func(scores, ground_truth)
    res = f"{label}: {r:.3f}"
    if include_pvalue:
        res += f" ({p=:.5f})"
    return res


def is_time_id_necessary(time_embedding_type):
    """Return true if the given time embedding type involves a time_id input"""
    return "attention" in time_embedding_type
