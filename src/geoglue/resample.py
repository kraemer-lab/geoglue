import os
import tempfile
import warnings
from pathlib import Path
from typing import Literal
from collections.abc import Iterator
from contextlib import contextmanager

import cdo
import xarray as xr
import numpy as np
from .util import is_lonlat
from .types import Bbox, CdoGriddes

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*MemoryRaster.*")

from .memoryraster import MemoryRaster  # noqa

WARN_BELOW_COVERAGE = 0.8


def remapbil_sparse(
    infile: str | Path,
    griddes_file: str,
    outfile: str | Path,
    eps: float = 1e-6,
    tmp_path: Path = Path("."),
) -> Path:
    """Sparse bilinear resampling

    Resampling a raster with standard CDO remapbil can cause issues such
    as NaNs moving into coastal regions for covariates where the variable
    is only defined on land (soil moisture, vegetation). This implementation
    uses a zero-filled resampled DataArray divided by a resampled mask
    (non-NA=1, NA=0). A low epsilon threshold is used to small contributions
    to avoid blowing up output near edges with NaN cells.

    Parameters
    ----------
    infile
        Input data file or xarray.DataArray
    griddes
        Target griddes file
    outfile
        Output resampled file path, if not specified, generated from infile by
    eps
        epsilon value that is used as a threshold for mask
    tmp_path
        Temporary folder to use for intermediate files, defaults to $CWD

    Returns
    -------
    xr.DataArray
        Returns sparse resampled DataArray
    """

    _cdo = cdo.Cdo()
    infile = Path(infile)
    da = xr.open_dataarray(infile)

    fill = da.fillna(0)
    mask = da.where(da.isnull(), 1).fillna(0)

    fill.to_netcdf(infile_fill := tmp_path / f"{infile.stem}_fill.nc")
    mask.to_netcdf(infile_mask := tmp_path / f"{infile.stem}_mask.nc")
    outfile_fill = tmp_path / f"{infile.stem}_remapbil_fill.nc"
    outfile_mask = tmp_path / f"{infile.stem}_remapbil_mask.nc"

    if not outfile_fill.exists():
        _cdo.remapbil(griddes_file, input=str(infile_fill), output=str(outfile_fill))
    if not outfile_mask.exists():
        _cdo.remapbil(griddes_file, input=str(infile_mask), output=str(outfile_mask))

    resamp_fill = xr.open_dataarray(outfile_fill)
    resamp_mask = xr.open_dataarray(outfile_mask)

    valid = resamp_mask > eps
    resamp = xr.where(valid, resamp_fill / resamp_mask, np.nan)
    resamp.encoding["_FillValue"] = da.encoding["_FillValue"]
    resamp.to_netcdf(outfile)
    return Path(outfile)


def resample(
    resampling: Literal["remapbil", "remapdis", "sremapbil"],
    infile: str | Path,
    target: MemoryRaster | CdoGriddes,
    outfile: str | Path | None = None,
    skip_exists=True,
) -> Path:
    """Resamples input file to output file using CDO's resampling to a target raster grid

    Parameters
    ----------
    resampling
        Resampling type to use, must be one of `remapbil`, `remapdis` or `sremapbil`:

        - `remapbil` is bilinear resampling
        - `remapdis` is distance-weighted average remapping
        - `sremapdis` is sparse bilinear resampling that uses a non-NaN/NaN mask
           to normalise values to avoid NaN spreading from land-ocean boundaries
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
    raster_bbox = Bbox.from_xarray(xr.open_dataset(infile, decode_timedelta=True))
    target_bbox = target.bbox if isinstance(target, MemoryRaster) else target.get_bbox()
    if not raster_bbox > target_bbox:
        warnings.warn(f"""
Raster bbox should entirely cover target bbox to ensure no NA in CDO resample.
If you cropped the raster from a larger one, you can use Bbox.enlarge() to
increase the bbox size so that it covers the target raster or grid.

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
            case "sremapbil":
                with tempfile.TemporaryDirectory(prefix="geoglue-") as tmp:
                    return remapbil_sparse(
                        infile, griddes.name, outfile, tmp_path=Path(tmp)
                    )

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
