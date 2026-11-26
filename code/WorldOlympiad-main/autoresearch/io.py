"""Canonical filesystem utilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: str | Path, model: type[T]) -> T:
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return output
