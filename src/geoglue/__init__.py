import os
from pathlib import Path

from .memoryraster import MemoryRaster
from .region import Region

if data_home := os.getenv("XDG_DATA_HOME"):
    data_path = Path(data_home) / "geoglue"
else:
    data_path = Path.home() / ".local" / "share" / "geoglue"
if not data_path.exists():
    data_path.mkdir(parents = True)

__all__ = ["MemoryRaster", "Region"]
