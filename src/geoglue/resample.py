import os
import tempfile
import warnings
from pathlib import Path
from typing import Literal
from collections.abc import Iterator
from contextlib import contextmanager

import cdo
import xarray as xr
from netCDF4 import Dataset

from .util import is_lonlat, sha256
from .types import Bbox, CdoGriddes

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*MemoryRaster.*")

from .memoryraster import MemoryRaster  # noqa

WARN_BELOW_COVERAGE = 0.8


def resample(
    resampling: Literal["remapbil", "remapdis"],
    infile: str | Path,
    target: MemoryRaster | CdoGriddes,
    outfile: str | Path | None = None,
    skip_exists=True,
) -> Path:
    """Resamples input file to output file using CDO's resampling to a target raster grid

    Parameters
    ----------
    resampling
        Resampling type to use, must be one of `remapbil` or `remapdis`
    infile
        Input file to read
    target
        Target MemoryRaster whose grid to resample to, or a CdoGriddes
    outfile
        Output resampled file path, if not specified, generated from infile by
        affixing `.resampled` to the path
    skip_exists
        Whether to skip resampling if outfile exists (default=True)

    Returns
    -------
    Path
        Resampled dataset path
    """

    _cdo = cdo.Cdo()
    if isinstance(infile, str):
        infile = Path(infile)
    infile_hash = sha256(infile, prefix=True)
    raster_bbox = Bbox.from_xarray(xr.open_dataset(infile, decode_timedelta=True))
    target_bbox = target.bbox if isinstance(target, MemoryRaster) else target.get_bbox()
    if (coverage := raster_bbox.coverage_fraction(target_bbox)) == 0:
        raise ValueError("No intersection between input raster and target")
    if coverage < WARN_BELOW_COVERAGE:
        warnings.warn(f"""
Insufficient overlap ({coverage:.1%}, expected {WARN_BELOW_COVERAGE:.0%}) between input raster
and target. CDO resample may result in (unintended) NA values in the output.
Consider using MemoryRaster.crop() or DataArray.sel() to match extents of input
raster and target raster; which should ideally only vary in grid cell size.

Raster bounds: {raster_bbox}
Target bounds: {target_bbox}
""")

    if not is_lonlat(infile):
        raise ValueError(
            "resample only supports lonlat grid, input file does not conform"
        )
    if (isinstance(target, MemoryRaster) and not target.is_lonlat) or (
        isinstance(target, CdoGriddes) and target.gridtype != "lonlat"
    ):
        raise ValueError("resample only supports lonlat grid, target does not conform")
    if outfile is None:
        outfile = infile.parent / f"{infile.stem}_{resampling}.nc"
    if Path(outfile).exists() and skip_exists:
        return Path(outfile)
    with tempfile.NamedTemporaryFile(suffix=".txt") as griddes:
        if isinstance(target, MemoryRaster):
            Path(griddes.name).write_text(str(target.griddes))
        else:
            Path(griddes.name).write_text(str(target))

        match resampling:
            case "remapbil":
                _cdo.remapbil(griddes.name, input=str(infile), output=str(outfile))
            case "remapdis":
                _cdo.remapdis(griddes.name, input=str(infile), output=str(outfile))
    # re-open file and add checksums of infile

    nc = Dataset(outfile, mode="a")
    nc.provenance = f"resample.infile={infile_hash} {infile}\n" + getattr(
        nc, "provenance", ""
    )
    nc.close()
    return Path(outfile)


@contextmanager
def resampled_dataset(
    resampling: Literal["remapbil", "remapdis"],
    data: str | Path | xr.Dataset,
    target: MemoryRaster,
) -> Iterator[xr.Dataset]:
    """Context manager version of :meth:`geoglue.resample.resample`.

    Parameters
    ----------
    resampling
        Resampling type to use, must be one of `remapbil` or `remapdis`
    data
        Input file to read or xarray dataset
    target
        Target MemoryRaster whose grid to resample to

    Yields
    ------
    xr.Dataset
        Resampled dataset

    Example
    -------
    >>> from geoglue.resample import resampled_dataset
    >>> from geoglue import MemoryRaster
    >>> pop = MemoryRaster.read("VNM_ppp_2000_1km_Aggregated_UNadj.tif")
    >>> with resampled_dataset("remapbil", "somefile.nc", pop) as ds:
    ...     print(ds)
    """
    if isinstance(data, xr.Dataset):
        # write to temporary file
        fd, data_path = tempfile.mkstemp(prefix="geoglue-", suffix=".nc")
        data.to_netcdf(data_path)  # type: ignore
        infile_istempfile = data_path, True
    else:
        infile_istempfile = Path(data), False

    with tempfile.NamedTemporaryFile(prefix="geoglue-", suffix=".nc") as f:
        resample(resampling, infile_istempfile[0], target, f.name, skip_exists=False)
        ds = xr.open_dataset(f.name, engine="netcdf4")
        yield ds

    if infile_istempfile[1]:  # clean up temporary file
        os.close(fd)  # type: ignore
        os.unlink(infile_istempfile[0])
