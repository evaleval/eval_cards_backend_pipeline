"""Read logical JSON bytes and write plain or deterministic gzip JSON files."""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json_bytes(path: Path) -> bytes:
    """Return uncompressed bytes so evidence hashes do not depend on storage."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def read_json(path: Path) -> Any:
    return json.loads(read_json_bytes(path))


def write_json_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish JSON bytes, with a stable gzip header when requested."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            if path.suffix == ".gz":
                with gzip.GzipFile(
                    filename="", mode="wb", compresslevel=9, fileobj=handle, mtime=0,
                ) as compressed:
                    compressed.write(payload)
            else:
                handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
