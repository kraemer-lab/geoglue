"Paths referred to by geoglue"

import os
from functools import cache
from pathlib import Path


@cache
def get_data_path() -> Path:
    if data_home := os.getenv("XDG_DATA_HOME"):
        data_path = Path(data_home) / "geoglue"
    else:
        data_path = Path.home() / ".local" / "share" / "geoglue"
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    return data_path


@cache
def get_cache_path() -> Path:
    if cache_home := os.getenv("XDG_CACHE_HOME"):
        cache_path = Path(cache_home) / "geoglue"
    else:
        cache_path = Path.home() / ".cache" / "geoglue"
    if not cache_path.exists():
        cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


geoglue_data_path = get_data_path()
geoglue_cache_path = get_cache_path()
