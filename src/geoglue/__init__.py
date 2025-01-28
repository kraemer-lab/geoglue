from .memoryraster import MemoryRaster
import os
from pathlib import Path

if data_home := os.getenv("XDG_DATA_HOME"):
    data_path = Path(data_home) / "geoglue"
else:
    data_path = Path.home() / ".local" / "share" / "geoglue"
if not data_path.exists():
    data_path.mkdir(parents = True)

__all__ = ["MemoryRaster"]
