"""Utilities for fetching ImageNet pretrained backbone weights on first use."""
from __future__ import annotations

import os
from urllib.request import urlretrieve


def ensure_weight(url: str, dest: str) -> str:
    """Download `url` to `dest` if `dest` does not already exist.

    Returns the absolute path to the (possibly just-downloaded) file. The
    parent directory is created on demand so the destination can live
    under any path the caller chooses. Already-present files are left
    untouched, including their mtime, so the function is safe to call on
    every training run.
    """
    dest_abs = os.path.abspath(dest)
    if os.path.exists(dest_abs) and os.path.getsize(dest_abs) > 0:
        return dest_abs
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    tmp = dest_abs + ".part"
    print(f"[weights] downloading {url}\n[weights]   -> {dest_abs}")
    try:
        urlretrieve(url, tmp)
        os.replace(tmp, dest_abs)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dest_abs
