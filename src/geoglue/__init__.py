import os
from pathlib import Path

from .memoryraster import MemoryRaster
from .region import Region, get_region

if data_home := os.getenv("XDG_DATA_HOME"):
    data_path = Path(data_home) / "geoglue"
else:
    data_path = Path.home() / ".local" / "share" / "geoglue"
if not data_path.exists():
    data_path.mkdir(parents=True)

if cache_home := os.getenv("XDG_CACHE_HOME"):
    cache_path = Path(cache_home) / "geoglue"
else:
    cache_path = Path.home() / ".cache" / "geoglue"
if not cache_path.exists():
    cache_path.mkdir(parents=True)

__all__ = ["MemoryRaster", "Region", "get_region"]
