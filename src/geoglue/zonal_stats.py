"""
Perform zonal statistics using exactextract
"""

import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from .util import find_unique_time_coord
from .memoryraster import MemoryRaster

warnings.warn(
    "geoglue.zonal_stats is deprecated and will be removed in geoglue in its initial release, use geoglue.zonalstats instead",
    FutureWarning,
    stacklevel=2,
)


def zonal_stats(
    da: xr.DataArray,
    geom: gpd.GeoDataFrame,
    operation: str = "mean(coverage_weight=area_spherical_km2)",
    weights: MemoryRaster | None = None,
    include_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return zonal statistics for a particular data array.

    Note that this function does not perform certain pre-processing steps,
    as they may not be required in general. See the functions mentioned below
    for more information.

    Parameters
    ----------
    da : xr.DataArray
        xarray DataArray to perform zonal statistics on. Must have
        'latitude', 'longitude' and a time coordinate
    geom : gpd.GeoDataFrame
        DataFrame containing a geometry column specifying the zones
        over which to calculate statistics
    operation : str
        Zonal statistics operation. For a full list of operations, see
        https://isciences.github.io/exactextract/operations.html. Default
        operation is to calculate the mean with a spherical area coverage weight.
    weights : MemoryRaster | None
        Optional, if specified, uses the specified raster to perform weighted
        zonal statistics.
    include_cols : list[str] | None
        Optional, if specified, only includes these columns. If not specified,
        returns all columns except the geometry column

    Returns
    -------
    pd.DataFrame
        The DataFrame specified by the `geom` parameter, one additional column,
        `value` containing the zonal statistic for the corresponding geometry.

    See Also
    --------
    zonal_stats_xarray
        Version of this function that returns a xarray DataArray
    geoglue.util.sort_lonlat
        Function to sort latitude and longitude
    geoglue.util.crop_dataset_to_geometry
        Function to crop dataset to geometry if dataset and geometry do not match
    """
    include_cols = (
        [c for c in geom.columns if c != "geometry"]
        if include_cols is None
        else include_cols
    )
    assert set(include_cols) < set(
        geom.columns
    ), f"{include_cols=} specifies columns not present in the geometry dataframe"
    time_coord = find_unique_time_coord(da)
    # shape after dropping time axis should be identical
    if operation.startswith("area_weighted_sum") and weights is None:
        warnings.warn(
            "Area weighted sum passed without weights is equivalent\n"
            "to using mean with the coverage weight set to area, using:\n"
            'operation="mean(coverage_weight=area_spherical_km2)"'
        )
        operation = "mean(coverage_weight=area_spherical_km2)"
    if weights:
        if da.shape[1:] != weights.shape:
            raise ValueError(
                f"Variable shape {da.shape[1:]} and weights shape {weights.shape} must be identical"
            )
        if "weighted" not in operation:
            operation = "weighted_" + operation
        weights = weights.mask(geom).astype(np.float32)

    exactextract_output_column = (
        re.match(r"(\w+)(?=\()", operation).group(1)  # type: ignore
        if "(" in operation
        else operation
    )
    out = None
    for date in da.coords[time_coord].values:
        rast = MemoryRaster.from_xarray(da.sel({time_coord: date}))
        if weights:
            if not operation.startswith("area_weighted_sum"):
                df = rast.zonal_stats(
                    geom,
                    operation,
                    weights=weights,
                    include_cols=include_cols,
                ).rename(columns={exactextract_output_column: "value"})
            else:
                df = rast.zonal_stats(
                    geom,
                    [
                        "weighted_sum(coverage_weight=area_spherical_km2)",
                        "count(coverage_weight=area_spherical_km2)",
                    ],
                    weights=weights,
                    include_cols=include_cols,
                )
                df["value"] = df["weighted_sum"] / df["count"]
        else:
            df = rast.zonal_stats(
                geom,
                operation,
                include_cols=include_cols,
            ).rename(columns={exactextract_output_column: "value"})

        df["date"] = date
        out = df if out is None else pd.concat([out, df])
    return out


def zonal_stats_xarray(
    da: xr.DataArray,
    geom: gpd.GeoDataFrame,
    operation: str = "mean(coverage_weight=area_spherical_km2)",
    weights: MemoryRaster | None = None,
    region_col: str | None = None,
) -> xr.DataArray:
    """Return zonal statistics for a DataArray.

    Note that this function does not perform certain pre-processing steps,
    as they may not be required in general. See the functions mentioned below
    for more information.


    Parameters
    ----------
    da : xr.DataArray
        xarray DataArray to perform zonal statistics on. Must have
        'latitude', 'longitude' and a time coordinate
    geom : gpd.GeoDataFrame
        DataFrame containing a geometry column specifying the zones
        over which to calculate statistics
    operation : str
        Zonal statistics operation. For a full list of operations, see
        https://isciences.github.io/exactextract/operations.html. Default
        operation is to calculate the mean with a spherical area coverage weight.
    weights : MemoryRaster | None
        Optional, if specified, uses the specified raster to perform weighted
        zonal statistics.
    region_col : str | None
        Column to use as elements of the `region` coordinate, optional. If not
        specified, is set to the first column in the geometry that has unique
        values for each row.

    Returns
    -------
    xr.DataArray
        DataArray with `region` and `date` as coordinates

    See Also
    --------
    zonal_stats
        Version of this function that returns a DataFrame
    geoglue.util.sort_lonlat
        Function to sort latitude and longitude
    geoglue.util.crop_dataset_to_geometry
        Function to crop dataset to geometry if dataset and geometry do not match
    """
    if region_col:
        assert region_col in geom.columns, f"Geometry should have {region_col=}"
    else:
        # pick a column which is different for each record
        N = len(geom)
        uniq_counts = [
            (c, len(geom[c].unique())) for c in geom.columns if c != "geometry"
        ]
        uniq_counts = [(c, n) for c, n in uniq_counts if n == N]
        if not uniq_counts:
            raise ValueError("No column found in geometry that can be primary key")
        else:
            region_col = uniq_counts[0][0]
    df = zonal_stats(da, geom, operation, weights, include_cols=[region_col])  # type: ignore
    df = df.rename(columns={region_col: "region"})
    pivoted = df[["region", "date", "value"]].pivot(
        index="date", columns="region", values="value"
    )
    da = xr.DataArray(
        data=pivoted.values,
        coords={"date": pivoted.index, "region": pivoted.columns},
        dims=["date", "region"],
    )
    da.coords["region"].attrs["original_name"] = region_col
    return da
