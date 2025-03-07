import tempfile
from pathlib import Path
from typing import Literal

import cdo

from .memoryraster import MemoryRaster
from .util import is_lonlat

_cdo = cdo.Cdo()

def resample(
    resampling: Literal["remapbil", "remapdis"],
    infile: str | Path,
    target: MemoryRaster,
    outfile: str | Path | None = None,
    skip_exists=True,
) -> Path:
    "Resamples input file to output file using CDO's resampling to a target raster grid"
    if isinstance(infile, str):
        infile = Path(infile)
    if not is_lonlat(infile):
        raise ValueError("resample only supports lonlat grid, input file does not conform")
    if not target.is_lonlat:
        raise ValueError("resample only supports lonlat grid, target MemoryRaster does not conform")
    if outfile is None:
        outfile = infile.parent / f"{infile.stem}_{resampling}.nc"
    if Path(outfile).exists() and skip_exists:
        return Path(outfile)
    with tempfile.NamedTemporaryFile(suffix=".txt") as griddes:
        Path(griddes.name).write_text(str(target.griddes))

        match resampling:
            case "remapbil":
                _cdo.remapbil(griddes.name, input=str(infile), output=str(outfile))
            case "remapdis":
                _cdo.remapdis(griddes.name, input=str(infile), output=str(outfile))
        return Path(outfile)
