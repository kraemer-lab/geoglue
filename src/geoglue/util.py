"Utility functions for geoglue"

import logging
from pathlib import Path
import shutil

import xarray as xr
import requests
import pandas as pd
import geopandas as gpd
import numpy as np


COMPRESSED_FILE_EXTS = [".tar.gz", ".tar.bz2", ".zip"]


def set_lonlat_attrs(ds: xr.Dataset):
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


def find_time_coords(ds: xr.Dataset) -> list[str]:
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


def find_unique_time_coord(ds: xr.Dataset) -> str:
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
    return {"longitude", "latitude"} < set(ds.coords)


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
            logging.info(f"Unpacking downloaded file {path}")
            unpack_file(path, unpack_in_folder)
        return True
    else:
        logging.error(f"Failed to fetch {url}, status={r.status_code}")
    return False


# Plotting functions
def geom_plot(df: pd.DataFrame, geometry: gpd.GeoDataFrame, col: str = "value"):
    return gpd.GeoDataFrame(df.merge(geometry)).plot(col)
