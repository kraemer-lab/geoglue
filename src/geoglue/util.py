"Utility functions for geoglue"

import logging
from pathlib import Path
import shutil

import xarray as xr
import requests
import pandas as pd
import geopandas as gpd


COMPRESSED_FILE_EXTS = [".tar.gz", ".tar.bz2", ".zip"]


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
