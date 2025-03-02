"""
ERA5 data download and data aggregation
---------------------------------------

This module downloads ERA5 daily level data and provides classes to
resample and aggregate data to administrative levels obtained from GADM
"""

from __future__ import annotations

import re
import datetime
import tempfile
import warnings
from pathlib import Path
from functools import cache

import cdo
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

from .util import get_extents
from .types import CdoGriddes
from .memoryraster import MemoryRaster

_cdo = cdo.Cdo()


class DatasetZonalStatistics:
    """Calculates zonal statistics for a daily temporal dataset"""

    def __init__(
        self,
        dataset: xr.Dataset,
        geom: gpd.GeoDataFrame,
        weights: MemoryRaster | None = None,
        include_cols: list[str] | None = None,
        time_col: str | None = None,
    ):
        self.dataset = dataset
        self.geom = geom
        self.weighted = weights is not None
        if time_col is None:
            matching_cols = [
                c
                for c in map(str, self.dataset.variables)
                if c.endswith("time") or c.startswith("time") or c == "t"
            ]
            if matching_cols:
                self.time_col = matching_cols[0]
            else:
                raise ValueError("No time_col supplied and none found by string match")
        else:
            assert (
                time_col in self.dataset.variables
            ), f"{time_col=} not found in dataset variables"
            self.time_col = time_col
        self.include_cols = (
            [c for c in self.geom.columns if c != "geometry"]
            if include_cols is None
            else include_cols
        )
        assert set(self.include_cols) < set(
            geom.columns
        ), f"{include_cols=} specifies columns not present in the geometry dataframe"

        # auto-fix longitude 0 -- 360
        if float(self.dataset.coords["longitude"].max()) > 180:
            self.dataset.coords["longitude"] = (
                self.dataset.coords["longitude"] + 180
            ) % 360 - 180
            self.dataset = self.dataset.sortby(self.dataset.longitude)
        self.dataset = self.dataset.sortby(self.dataset.latitude, ascending=False)
        # Crop data to geometry extents
        extent_long, extent_lat = get_extents(self.geom)
        self.dataset = self.dataset.sel(longitude=extent_long, latitude=extent_lat)

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

    def to_netcdf(self, path: str, fix_griddes: bool = True):
        "Save NetCDF data"
        if not fix_griddes:
            self.dataset.to_netcdf(path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".nc") as f:
                self.dataset.to_netcdf(f.name)
                griddes = CdoGriddes.from_file(f.name)
                if griddes.gridtype == "generic":
                    griddes.gridtype = "lonlat"
                    with tempfile.NamedTemporaryFile(suffix=".txt") as grid_tmp:
                        Path(grid_tmp.name).write_text(str(griddes))
                        _cdo.setgrid(grid_tmp.name, input=f.name, output=path)
            # verify griddes was fixed
            assert CdoGriddes.from_file(path).gridtype == "lonlat"

    @cache
    def __getitem__(self, variable: str) -> xr.DataArray:
        return self.dataset[variable]

    def zonal_stats(
        self,
        variable: str,
        operation: str = "mean(coverage_weight=area_spherical_km2)",
        weighted: bool = True,
        min_date: datetime.date | None = None,
        max_date: datetime.date | None = None,
        const_cols: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        da = self[variable]
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
        min_date = min_date or da[self.time_col].min().dt.date.item(0)
        max_date = max_date or da[self.time_col].max().dt.date.item(0)
        assert max_date >= min_date, "End date must be later than start date"  # type: ignore

        exactextract_output_column = (
            re.match(r"(\w+)(?=\()", operation).group(1)  # type: ignore
            if "(" in operation
            else operation
        )
        out = pd.DataFrame(data=[], columns=self.include_cols + ["value", "date"])

        for date in tqdm(pd.date_range(min_date, max_date, inclusive="both")):
            rast = MemoryRaster.from_xarray(da.sel({self.time_col: date}))
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
