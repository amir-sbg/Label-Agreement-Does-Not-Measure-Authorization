from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "NEUROMARK_ROOT"


def data_root() -> Path:
    value = os.environ.get(ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"Set {ROOT_ENV} to the directory that holds the COBRE/FBIRN Data/ and Results/ trees."
        )
    return Path(value)


def data_path(relative: str) -> str:
    return str(data_root() / relative)


def expand(value: str) -> str:
    if "${" + ROOT_ENV + "}" in value:
        value = value.replace("${" + ROOT_ENV + "}", str(data_root()))
    return value


def public_locator(locator: str) -> str:
    value = os.environ.get(ROOT_ENV)
    if not value:
        return locator
    root = str(Path(value))
    return locator.replace(f"{root}/", "neuromark:").replace(root, "neuromark")
