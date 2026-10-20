"""Retry wrapper for file writes.

Output lives on ``/mnt/d`` which, under WSL2, is a DrvFs mount that
intermittently raises ``OSError: [Errno 22] Invalid argument`` on ``open(...,
"w+b")`` when it is under heavy concurrent I/O (e.g. right after a long raster
streaming pass). The write succeeds on a retry a moment later.
"""
from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_io(fn: Callable[..., T] | None = None, *, tries: int = 6, delay: float = 1.5):
    def deco(f: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(f)
        def wrap(*args, **kwargs) -> T:
            last: OSError | None = None
            for attempt in range(tries):
                try:
                    return f(*args, **kwargs)
                except OSError as exc:  # EINVAL, transient EIO, etc.
                    last = exc
                    if attempt < tries - 1:
                        time.sleep(delay * (attempt + 1))
            raise last  # type: ignore[misc]

        return wrap

    return deco(fn) if fn is not None else deco


@retry_io
def write_bytes(path, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)


@retry_io
def write_text(path, text: str, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as fh:
        fh.write(text)


@retry_io
def save_fig(fig, path, **kwargs) -> None:
    fig.savefig(path, **kwargs)


@retry_io
def to_csv(df, path, **kwargs) -> None:
    df.to_csv(path, **kwargs)


@retry_io
def to_parquet(df, path, **kwargs) -> None:
    df.to_parquet(path, **kwargs)


@retry_io
def savez(path, **arrays) -> None:
    import numpy as np

    np.savez_compressed(path, **arrays)


@retry_io
def read_band(src, *args, **kwargs):
    """Retrying wrapper around ``rasterio`` dataset reads.

    The source rasters live on a DrvFs mount (an external drive, remounted
    manually at least once this project) that has shown transient
    ``TIFFReadEncodedStrip`` read failures under load, alongside the write-side
    EINVAL errors ``save_fig``/``to_csv``/etc. already retry around.
    """
    return src.read(*args, **kwargs)
