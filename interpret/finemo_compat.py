"""Run the finemo CLI on numpy < 1.24, which lacks the dtype argument of np.stack that finemo uses.

Usage: python interpret/finemo_compat.py <finemo subcommand> [args...]
"""

import sys

import numpy as np

if "dtype" not in np.stack.__doc__.split("Returns")[0]:
    _stack = np.stack

    def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
        result = _stack(arrays, axis=axis, out=out)
        return result if dtype is None else result.astype(dtype, casting=casting)

    np.stack = stack

from finemo.main import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli())
