"""Zonal stats compute

This is going to replace geoglue.zonal_stats
"""

import re
import logging
from pathlib import Path

import exactextract
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from .types import CdoGriddes
from .resample import resample
from .config import ZonalStatsConfig
from .util import read_geotiff

logger = logging.getLogger(__name__)


def _slice_extract_core(
    arr2d: np.ndarray,
    *,
    vec: gpd.GeoDataFrame,
    ops: str,
    weights: xr.DataArray | None,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """
    arr2d: 2D numpy array (latitude x longitude) supplied by apply_ufunc
    vec, ops, weights, lat, lon passed via kwargs
    Returns: 1D numpy array of length = number of regions
    """
    # rebuild an xarray.DataArray as we get a numpy array
    da = xr.DataArray(
        arr2d,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )
    if not ops.startswith("area_weighted_sum"):
        val: pd.DataFrame = exactextract.exact_extract(
            da, vec, ops, weights=weights, output="pandas"
        )  # type: ignore
    else:
        op_params = ""
        if "(" in ops and ")" in ops:
            # extract the bit between brackets
            if op_match := re.search(r"\((.*)\)", ops):
                op_params = "(" + op_match.group(1) + ")"
            else:
                raise ValueError(f"Invalid operation supplied: {ops}")
            logger.info(f"zonalstats using {op_params=}")
        res: pd.DataFrame = exactextract.exact_extract(
            da,
            vec,
            [f"weighted_sum{op_params}", f"count{op_params}"],
            weights=weights,
            output="pandas",
        )  # type: ignore
        val = res["weighted_sum"] / res["count"]
    return val.values.squeeze()


def zonalstats(
    rast: xr.DataArray,
    vec: gpd.GeoDataFrame,
    ops: str,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Run exactextract over the spatial core dims (latitude, longitude) in parallel
    across any other (non-spatial) dimensions of `rast`.
    Returns an xarray.DataArray whose new core dim is "region".
    """
    kwargs = {
        "vec": vec,
        "ops": ops,
        "weights": weights,
        "lat": rast["latitude"].values,
        "lon": rast["longitude"].values,
    }
    if ops.startswith("area_weighted_sum") and weights is None:
        raise ValueError("area_weighted_sum requires weights to be set")

    result = xr.apply_ufunc(
        _slice_extract_core,
        rast,
        input_core_dims=[["latitude", "longitude"]],
        output_core_dims=[["region"]],
        vectorize=True,  # vectorize across non-core dims (e.g. time, month)
        kwargs=kwargs,
    )
    return result.assign_coords(region=("region", vec.index))


def compute_config(cfg: ZonalStatsConfig) -> xr.DataArray:
    # try reading files
    cfg.check_exists()
    if cfg.raster.suffix != ".nc":
        raise ValueError("Unsupported file format %r", cfg.raster)
    raster_path = cfg.raster
    vec: gpd.GeoDataFrame = gpd.read_file(cfg.shapefile).set_index(cfg.shapefile_id)  # type: ignore
    if not cfg.output.parent.exists():
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
    if cfg.weights is None:
        weights = None
    elif cfg.weights.suffix in [".tif", ".tiff"]:
        weights = read_geotiff(cfg.weights)
    else:
        weights = xr.open_dataarray(cfg.weights)
    resampled_path: Path = raster_path.parent / f"{cfg.raster.stem}.{cfg.resample}.nc"

    # Resample raster to weights unless resampling = 'off'
    if weights is not None and cfg.resample != "off":
        griddes = CdoGriddes.from_dataset(weights)
        assert griddes.gridtype == "lonlat"
        print(f"resample\t{cfg.raster} -> {resampled_path}")
        raster_path = resample(cfg.resample, raster_path, griddes, resampled_path)

    # At this point, raster and weights grid should be aligned (if using weights)
    # Run exactextract in parallel using dask across non-spatial dimensions
    # Currently only dataarrays are supported
    rast = xr.open_dataarray(raster_path)
    da = zonalstats(rast, vec, cfg.operation, weights=weights)
    da.attrs["geoglue_config"] = str(cfg)
    return da
