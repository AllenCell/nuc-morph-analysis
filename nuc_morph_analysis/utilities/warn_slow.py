import datetime
from functools import wraps
import time
from typing import Callable, Union

import pandas as pd


def warn_slow(threshold: Union[pd.Timedelta, datetime.timedelta, int, str]):
    """
    Decorator to print a warning if the decorated function takes longer than `threshold` seconds
    to complete.

    Example
    -------
    >>> from nuc_morph_analysis.utilities.warn_slow import warn_slow
    >>> import time
    >>> @warn_slow("5m30s")
    ... def foo():
    ...     time.sleep(6)
    ...     return "foo"
    ...
    >>> foo()
    Warning: foo was slower than expected (6.01s)
    'foo'

    Parameters
    ----------
    threshold: pandas.Timedelta, datetime.timedelta, int, or str
        How long the function should take to complete. If int, interpreted as seconds. String values
        are parsed by pd.Timedelta.
    """
    if type(threshold) == int:
        threshold = pd.Timedelta(seconds=threshold)
    else:
        threshold = pd.Timedelta(threshold)

    # Could simplify this a little with @pamda.curry()
    def _warn_slow(func: Callable):
        @wraps(func)
        def warn_slow_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            if pd.Timedelta(seconds=elapsed) > threshold:
                print(f"Warning: {func.__name__} was slower than expected ({elapsed:.2f}s)")
            return result

        return warn_slow_wrapper

    return _warn_slow
