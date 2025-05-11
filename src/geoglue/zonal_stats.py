"""
ERA5 data download and data aggregation
---------------------------------------

This module downloads ERA5 daily level data and provides classes to
resample and aggregate data to administrative levels obtained from GADM
"""

from __future__ import annotations

import re
import datetime
import warnings
from functools import cache

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from .util import get_extents, set_lonlat_attrs, find_unique_time_coord
from .memoryraster import MemoryRaster


class DatasetZonalStatistics:
    """Zonal statistics calculation for a daily temporal dataset

    Parameters
    ----------
    dataset
        Dataset to calculate zonal statistics for
    geom
        DataFrame containing a geometry column specifying the zones
        over which to calculate statistics
    weights
        Optional. If specified, uses the specified MemoryRaster as a
        weighting raster for zonal statistics calculations
    include_cols
        Optional. Columns from the `geom` DataFrame that are included
        in zonal statistics output. If not specified, includes all columns
        except the `geometry` column.
    time_coord
        Optional, time coordinate. If not specified, is detected
        automatically from the data types of the dataset coordinates
    crop_dataset_to_geometry
        Whether to crop dataset to geometry, default=False.

        Setting this parameter to True can give mismatches in shape with
        weights and is unnecessary if the original dataset matches weights
        extents via a cdo resample. This is because geoglue.resample does
        not take geometry into account.
    """
    def __init__(
        self,
        dataset: xr.Dataset,
        geom: gpd.GeoDataFrame,
        weights: MemoryRaster | None = None,
        include_cols: list[str] | None = None,
        time_coord: str | None = None,
        crop_dataset_to_geometry: bool = False,
    ):
        self.dataset = dataset
        self.geom = geom
        self.weighted = weights is not None
        self.time_coord = time_coord or find_unique_time_coord(self.dataset)
        self.include_cols = (
            [c for c in self.geom.columns if c != "geometry"]
            if include_cols is None
            else include_cols
        )
        assert set(self.include_cols) < set(geom.columns), (
            f"{include_cols=} specifies columns not present in the geometry dataframe"
        )

        # auto-fix longitude 0 -- 360
        if float(self.dataset.coords["longitude"].max()) > 180:
            self.dataset.coords["longitude"] = (
                self.dataset.coords["longitude"] + 180
            ) % 360 - 180
            self.dataset = self.dataset.sortby(self.dataset.longitude)
        self.dataset = self.dataset.sortby(self.dataset.latitude, ascending=False)

        # Crop data to geometry extents [off by default, see note in docstring]
        if crop_dataset_to_geometry:
            extent_long, extent_lat = get_extents(self.geom)
            self.dataset = self.dataset.sel(longitude=extent_long, latitude=extent_lat)
            set_lonlat_attrs(self.dataset)  # makes dataset cdo compatible

        if weights:
            self.weights = weights.mask(self.geom).astype(np.float32)
        else:
            self.weights = None
        self._variables: list[str] = [
            v for v in self.dataset.variables if v not in self.dataset.coords
        ]  # type: ignore

    @property
    def variables(self) -> list[str]:
        return self._variables

    def to_netcdf(self, *args, **kwargs):
        "Save NetCDF data"
        self.dataset.to_netcdf(*args, **kwargs)

    @cache
    def __getitem__(self, variable: str) -> xr.DataArray:
        return self.dataset[variable]

    def zonal_stats(
        self,
        variable: str,
        operation: str = "mean(coverage_weight=area_spherical_km2)",
        weighted: bool | None = None,
        min_date: datetime.date | None = None,
        max_date: datetime.date | None = None,
        const_cols: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Return zonal statistics for a particular variable in the dataset

        Parameters
        ----------
        variable : str
            Variable to perform zonal statistics on
        operation : str
            Zonal statistics operation. For a full list of operations, see
            https://isciences.github.io/exactextract/operations.html. Default
            operation is to calculate the mean with a spherical area coverage weight.
        weighted : bool | None
            Optional, whether to perform a weighted operation. If True, the weights parameter
            must be passed to DatasetZonalStatistics. If set to None (the default), a
            weighted operation is performed if the weights parameter is set; in that case
            the non-weighted zonal statistic can be returned by setting `weighted=False`
        min_date : datetime.date | None
            If specified, uses this as the start date when performing zonal statistics,
            otherwise performs zonal statistics from the beginning of the dataset.
        max_date : datetime.date | None
            If specified, uses this as the end date when performing zonal statistics,
            otherwise performs zonal statistics to the end of the dataset.
        const_cols : dict[str, str] | None
            A dictionary of key value pairs, optional, that are used to add
            columns with a constant value to the returned DataFrame. This can
            be used to add useful metadata, such as ISO3 code or data units.

        Returns
        -------
        pd.DataFrame
            The DataFrame specified by the `geom` parameter, with the included columns
            specified in `include_cols`, one additional column, `value` containing the
            zonal statistic for the corresponding geometry, and additional columns
            specified in `const_cols`
        """
        da = self.dataset[variable]
        weighted = weighted or self.weighted
        # shape after dropping time axis should be identical
        if operation.startswith("area_weighted_sum") and not weighted:
            warnings.warn(
                "Area weighted sum passed without weights is equivalent\n"
                "to using mean with the coverage weight set to area, using:\n"
                'operation="mean(coverage_weight=area_spherical_km2)"'
            )
            operation = "mean(coverage_weight=area_spherical_km2)"
        if weighted:
            if self.weights is None:
                raise ValueError(
                    "Weighted zonal statistics requested but no weights supplied"
                )
            if da.shape[1:] != self.weights.shape:
                raise ValueError(
                    f"Variable shape {da.shape[1:]} and weights shape {self.weights.shape} must be identical"
                )
            if "weighted" not in operation:
                operation = "weighted_" + operation
        min_date = min_date or da[self.time_coord].min().dt.date.item(0)
        max_date = max_date or da[self.time_coord].max().dt.date.item(0)
        assert max_date >= min_date, "End date must be later than start date"  # type: ignore

        exactextract_output_column = (
            re.match(r"(\w+)(?=\()", operation).group(1)  # type: ignore
            if "(" in operation
            else operation
        )
        out = pd.DataFrame(data=[], columns=self.include_cols + ["value", "date"])

        min_datetime64 = np.datetime64(min_date.isoformat())
        max_datetime64 = np.datetime64(max_date.isoformat() + "T23:59:59")
        times_in_range = [
            t
            for t in self.dataset.coords[self.time_coord].values
            if min_datetime64 <= t <= max_datetime64
        ]
        for date in times_in_range:
            rast = MemoryRaster.from_xarray(da.sel({self.time_coord: date}))
            if weighted:
                if not operation.startswith("area_weighted_sum"):
                    df = rast.zonal_stats(
                        self.geom,
                        operation,
                        weights=self.weights,
                        include_cols=self.include_cols,
                    ).rename(columns={exactextract_output_column: "value"})
                else:
                    df = rast.zonal_stats(
                        self.geom,
                        [
                            "weighted_sum(coverage_weight=area_spherical_km2)",
                            "count(coverage_weight=area_spherical_km2)",
                        ],
                        weights=self.weights,
                        include_cols=self.include_cols,
                    )
                    df["value"] = df["weighted_sum"] / df["count"]
            else:
                df = rast.zonal_stats(
                    self.geom,
                    operation,
                    include_cols=self.include_cols,
                ).rename(columns={exactextract_output_column: "value"})

            df["date"] = date
            out = pd.concat([out, df])
        if const_cols:
            for c, v in const_cols.items():
                out[c] = v
        return out
