"Utility functions for geoglue"

import re
import logging
import hashlib
import datetime
from pathlib import Path
from typing import TypeVar
import shutil

import xarray as xr
import requests
import pandas as pd
import geopandas as gpd
import numpy as np

from geoglue.types import Bbox

logger = logging.getLogger(__name__)

COMPRESSED_FILE_EXTS = [".tar.gz", ".tar.bz2", ".zip"]

X = TypeVar("X", xr.DataArray, xr.Dataset)


def read_raster(file: str | Path) -> xr.Dataset | xr.DataArray:
    match Path(file).suffix:
        case ".nc":
            rast = xr.open_dataset(file)
            return fix_lonlat(rast)
        case ".tif":
            rast = read_geotiff(file)
            return fix_lonlat(rast)
        case _:
            raise ValueError(f"Unsupported file format: {file}")


def bbox_from_region(region: str, integer_bounds: bool = False) -> Bbox:
    if "," in region:
        # try processing as a bbox
        bbox = Bbox.from_string(region)
    elif region.endswith(".shp"):  # crop to shapefile
        vec = gpd.read_file(region)
        bbox = Bbox(*vec.total_bounds)
    elif region.endswith(".nc") or region.endswith(".tif"):
        rast = read_raster(region)
        bbox = Bbox.from_xarray(rast)
    else:
        raise ValueError("Unsupported region, must be one of .shp, .tif, .nc or a bbox")
    return bbox if not integer_bounds else bbox.int()


def fix_lonlat(ds: X) -> X:
    if ds.longitude.max() > 180:
        ds = sort_lonlat(ds)
    set_lonlat_attrs(ds)
    return ds


def read_geotiff(path: str | Path) -> xr.DataArray:
    da = xr.open_dataarray(path).squeeze()
    if "x" in da.coords and "y" in da.coords:
        da = da.rename({"x": "longitude", "y": "latitude"})
    return fix_lonlat(da)


def logfmt_escape(value: str | Path | None) -> str:
    """
    Escape a string for logfmt-safe output

    Examples:
        logfmt_escape("ok") -> "ok"
        logfmt_escape("has space") -> "\"has space\""
        logfmt_escape("weird=\"val\"") -> "\"weird=\\\"val\\\"\""
    """
    if value is None:
        return '""'

    s = str(value)
    # check if quoting is needed
    if re.search(r'[\s="\\]', s):
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    return s


def write_variables(ds: xr.Dataset, path: Path) -> list[Path]:
    folder = path.parent
    out = []
    for var in ds.data_vars:
        var_path = folder / f"{path.stem}.{var}.nc"
        ds[var].to_netcdf(var_path)
        out.append(var_path)
    return out


def get_first_monday(year: int) -> datetime.date:
    "Gets first Monday of the year"
    return datetime.datetime.strptime(f"{year}-W01-1", "%Y-W%W-%u").date()


def get_last_sunday(year: int) -> datetime.date:
    "Gets last Sunday of year"
    d = datetime.datetime.strptime(f"{year}-W51-7", "%Y-W%W-%u").date()
    if (d + datetime.timedelta(days=7)).year == year:  # one more week in the year!
        return d + datetime.timedelta(days=7)
    else:
        return d


def sha256(file_path: str | Path, prefix: bool = False) -> str:
    """Returns SHA256 hash of a file

    Parameters
    ----------
    file_path : str | Path
        Path to file
    prefix : bool
        Whether to prefix 'sha256:' to the checksum, useful for disambiguating
        between multiple checksum methods

    Returns
    -------
    str
        Checksum with optional prefix
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    h = sha256_hash.hexdigest()
    return h if not prefix else "sha256:" + h


def crop_dataset_to_geometry(ds: xr.Dataset, geom: gpd.GeoDataFrame) -> xr.Dataset:
    """Crop xarray dataset to geometry

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to crop
    geom : gpd.GeoDataFrame
        Geometry to crop

    Returns
    -------
    xr.Dataset
        Cropped dataset
    """
    extent_long, extent_lat = get_extents(geom)
    ds = ds.sel(longitude=extent_long, latitude=extent_lat)
    set_lonlat_attrs(ds)  # makes dataset cdo compatible
    return ds


def sort_lonlat(ds: X) -> X:
    "Sorts longitude to -180 - 180 and latitude in descending order"
    if float(ds.coords["longitude"].max()) > 180:
        ds.coords["longitude"] = (ds.coords["longitude"] + 180) % 360 - 180
        ds = ds.sortby(ds.longitude)
    ds = ds.sortby(ds.latitude, ascending=False)
    set_lonlat_attrs(ds)
    return ds


def set_lonlat_attrs(ds: xr.Dataset | xr.DataArray):
    """Sets CF-compliant grid attributes

    Sets grid attributes so that cdo can recognise the grid as `lonlat`

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that is modified in place with the grid attributes. The
        dataset must have coordinates named either `lat`, `lon` or
        `latitude`, `longitude` to set grid attributes correctly
    """
    c_lat, c_lon = None, None
    if "lat" in ds.coords and "lon" in ds.coords:
        c_lat = "lat"
        c_lon = "lon"
    if "latitude" in ds.coords and "longitude" in ds.coords:
        c_lat = "latitude"
        c_lon = "longitude"
    if c_lat is None or c_lon is None:
        raise ValueError(
            "Could not determine latitude and longitude coords, must have either 'lon', 'lat' or 'longitude', 'latitude' columns"
        )
    ds.coords[c_lon].attrs.update(
        {
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "longitude",
        }
    )

    ds.coords[c_lat].attrs.update(
        {"units": "degrees_north", "standard_name": "latitude", "long_name": "latitude"}
    )


def find_time_coords(ds: xr.Dataset | xr.DataArray) -> list[str]:
    "Lists all time coordinates (dtype `np.datetime64`)"
    time_coords = []
    for coord in ds.coords:
        # Check if the data type is datetime64
        if np.issubdtype(ds[coord].dtype, np.datetime64):
            time_coords.append(coord)
        # Or check CF conventions
        elif "units" in ds[coord].attrs and "since" in ds[coord].attrs["units"]:
            time_coords.append(coord)
        elif ds[coord].attrs.get("standard_name") == "time":
            time_coords.append(coord)
    return time_coords


def find_unique_time_coord(ds: xr.Dataset | xr.DataArray) -> str:
    "Finds unique time coordinate or raises a ValueError"
    coords = find_time_coords(ds)
    match len(coords):
        case 0:
            raise ValueError("No time coordinate found")
        case 1:
            return coords[0]
        case _:
            raise ValueError(f"No unique time coordinate found: {coords}")


def zero_padded_intrange(start: int, end: int, inclusive=True) -> list[str]:
    assert end > start, "End of range must be higher than start of range"
    vals = range(start, end) if not inclusive else range(start, end + 1)
    n = len(str(end)) if not inclusive else len(str(end + 1))
    return [f"{i:0{n}d}" for i in vals]


def is_lonlat(data: str | Path | xr.Dataset | xr.DataArray) -> bool:
    if isinstance(data, (str, Path)):
        ds = xr.open_dataset(data)
    else:
        ds = data
    return {"longitude", "latitude"} <= set(ds.coords)


def get_extents(gdf: gpd.GeoDataFrame) -> tuple[slice, slice]:
    min_long, min_lat, max_long, max_lat = gdf.geometry.total_bounds
    return slice(int(min_long), int(max_long) + 1), slice(
        int(max_lat) + 1, int(min_lat)
    )


def unpack_file(path: Path, in_folder: Path | None = None):
    """Unpack a zipped file."""
    extract_dir = in_folder or path.parent
    shutil.unpack_archive(path, str(extract_dir))


def download_file(
    url: str, path: Path, unpack: bool = True, unpack_in_folder: Path | None = None
) -> bool:
    """Download a file from a given URL to a given path."""
    if path.is_dir():
        # get last bit of URL and append to folder
        path = path / url.split("/")[-1]
    if (r := requests.get(url)).status_code == 200:
        with open(path, "wb") as out:
            for bits in r.iter_content():
                out.write(bits)
        # Unpack file
        if unpack and any(str(path).endswith(ext) for ext in COMPRESSED_FILE_EXTS):
            logger.info(f"Unpacking downloaded file {path}")
            unpack_file(path, unpack_in_folder)
        return True
    else:
        logger.error(f"Failed to fetch {url}, status={r.status_code}")
    return False


# Plotting functions
def geom_plot(df: pd.DataFrame, geometry: gpd.GeoDataFrame, col: str = "value"):
    return gpd.GeoDataFrame(df.merge(geometry)).plot(col)
