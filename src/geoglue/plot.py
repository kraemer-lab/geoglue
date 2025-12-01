"""Plot data."""

import logging
from pathlib import Path

import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

from .config import read_zonalstats_config

logger = logging.getLogger(__name__)


def plot(
    da: xr.DataArray,
    isel: int | tuple[int],
    cmap: str = "viridis",
    output: str | None = None,
    geometry: str = ".",
):
    "Plot an xarray DataArray"

    geometry_path = Path(geometry)
    long_name = da.attrs.get("long_name")
    non_region_coords: list[str] = [
        str(c)
        for c in da.coords
        if c not in ["lat", "lon", "latitude", "longitude", "region"]
    ]
    if non_region_coords:
        if len(non_region_coords) == 1:
            isel_val = isel if isinstance(isel, int) else isel[0]
            isel_kwargs = {non_region_coords[0]: isel_val}
        else:
            assert isinstance(isel, tuple)
            if len(isel) != len(non_region_coords):
                raise ValueError(
                    "isel must be a tuple of indices representing the lat/lon raster to select"
                )
            isel_kwargs = dict(zip(non_region_coords, isel))
        print("Selecting", isel_kwargs)
        da = da.isel(**isel_kwargs)

    if "geoglue_config" in da.attrs:
        print("Detected zonalstats file with geoglue_config attribute")
        # try to read geometry from config
        cfg_str = da.attrs["geoglue_config"]
        cfg = read_zonalstats_config(cfg_str)
        if cfg is None:
            raise ValueError(f"Invalid geoglue_config attr found: {cfg_str!r}")
        if isinstance(cfg, list):  # pick the geometry from the first one
            shapefile = cfg[0].shapefile
        else:
            shapefile = cfg.shapefile
        if geometry_path.is_dir():
            # try loading shapefile relative to dir
            geom = gpd.read_file(geometry_path / shapefile)
        else:
            geom = gpd.read_file(geometry_path)

        geom[da.name] = da.values
        ax = geom.plot(da.name, cmap=cmap, legend=True)
        if long_name:
            ax.set_title(long_name)
    else:
        _, ax = plt.subplots()
        da.plot(cmap=cmap, ax=ax)  # type: ignore
        if long_name:
            ax.set_title(long_name)
        if geometry_path.exists() and not geometry_path.is_dir():
            gpd.read_file(geometry_path).boundary.plot(ax=ax)
    if output:
        plt.savefig(output)
    else:
        plt.show()
