# =============================================================
# utils.py
#
# Shared infrastructure helpers (not physical constants — see config.py).
# =============================================================

import os
import sys

OUTPUTS_ROOT = "outputs"


def get_output_dir(name=None):
    """
    Return outputs/<name>/, creating it if it doesn't exist.

    If `name` is omitted, it's inferred from the currently running
    script's filename — e.g. `python analysis/pypty_comparison.py`
    resolves to outputs/pypty_comparison/. Pass `name` explicitly to
    override (useful when called from a notebook or REPL, where
    sys.argv[0] isn't meaningful).
    """
    if name is None:
        name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    path = os.path.join(OUTPUTS_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path