"""
ERA5 data download and data aggregation
---------------------------------------

This module downloads ERA5 daily level data and provides classes to
resample and aggregate data to administrative levels obtained from GADM
"""

from __future__ import annotations

import datetime
from typing import Literal
from pathlib import Path
from functools import cache
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm import tqdm
from rasterio.enums import Resampling

from .gadm import GADM
from .memoryraster import MemoryRaster

ERA5_VARIABLE_MAPPINGS = {"2m_temperature": "t2m"}


def get_extents(gdf: gpd.GeoDataFrame) -> tuple[slice, slice]:
    min_long, min_lat, max_long, max_lat = gdf.geometry.total_bounds
    return slice(int(min_long), int(max_long) + 1), slice(
        int(max_lat + 1), int(min_lat)
    )


@dataclass
class ERA5Aggregated:
    data: pd.DataFrame
    geometry: gpd.GeoDataFrame
    admin_level: int
    temporal_scope: Literal["weekly", "daily"]

    def select(self, at: str):
        if self.temporal_scope == "weekly":
            return self.data[self.data.isoweek == at]
        else:
            at = datetime.date.fromisoformat(at)
            return self.data[self.data.date.dt.date == at]

    def plot(self, at: str):
        df = gpd.GeoDataFrame(self.select(at).merge(self.geometry))
        ax = df.plot("value", legend=True)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(df.metric.unique()[0])
        plt.show()

    def weekly(self) -> ERA5Aggregated:
        df = (
            self.data.groupby(GADM.list_admin_cols(self.admin_level) + ["isoweek"])
            .value.mean()
            .reset_index()
            .sort_values("isoweek")
        )
        df["metric"] += ".weekly_mean"
        return ERA5Aggregated(df, self.geometry, self.admin_level, "weekly")


class ERA5:
    def __init__(
        self,
        filename: str | Path,
        iso3: str,
        admin_level: int,
        statistic: Literal["daily_mean", "daily_max", "daily_min"],
    ):
        self.filename = filename
        self.statistic = statistic
        self.admin_level = admin_level
        self.iso3 = iso3
        self.geom = GADM(iso3)[admin_level]
        self.admin_cols = GADM.list_admin_cols(admin_level)
        self.dataset = xr.open_dataset(filename)
        self.variables = [v for v in self.dataset.variables]

    def __repr__(self):
        return (
            f"<ERA5 iso3={self.iso3} admin_level={self.admin_level} "
            f"statistic={self.statistic} filename={self.filename!r}>"
        )

    def set_population(self, raster: MemoryRaster):
        weather = self.get_variable("2m_temperature").isel(valid_time=0)
        self._population_original = raster
        _population_masked_high = self._population_original.mask(self.geom).astype(
            np.float32
        )
        # TODO: assumes weahter data is at a lower resolution than population
        self.population = _population_masked_high.resample(
            MemoryRaster.from_xarray(weather), Resampling.sum
        )

    @cache
    def get_variable(self, variable: str) -> xr.DataArray:
        varname = ERA5_VARIABLE_MAPPINGS.get(variable, variable)
        data = self.dataset[varname]

        # ERA5 stores longitude from the 0 to 360 scale
        # with 0 at the Greenwich meridian, counting east
        data.coords["longitude"] = (data.coords["longitude"] + 180) % 360 - 180
        data = data.sortby(data.longitude)
        # crop to geometry extent
        extent_long, extent_lat = get_extents(self.geom)
        return data.sel(longitude=extent_long, latitude=extent_lat)

    def zonal_daily(
        self,
        variable: str,
        operation: str = "weighted_mean(coverage_weight=area_spherical_m2)",
    ) -> ERA5Aggregated:
        da = self.get_variable(variable)
        min_date = da.valid_time.min().dt.date.item(0)
        # TODO: max_date = da.valid_time.max().dt.date.item(0)
        max_date = min_date

        # Empty dataframe with output columns
        out = pd.DataFrame(data=[], columns=self.admin_cols + ["value", "date"])  # type: ignore

        for date in tqdm(pd.date_range(min_date, max_date, inclusive="both")):
            arr = da.sel(valid_time=date.isoformat())
            rast = MemoryRaster.from_xarray(arr)
            df = rast.zonal_stats(
                self.geom,
                operation,
                weights=self.population,
                include_cols=self.admin_cols,
            ).rename(columns={"weighted_mean": "value"})
            df["date"] = date
            out = pd.concat([out, df])
        out["metric"] = f"era5.{variable}.{self.statistic}"
        out["isoweek"] = (
            out["date"].dt.isocalendar().year.astype(str)
            + "-W"
            + out["date"].dt.isocalendar().week.astype(str).str.zfill(2)
        )
        return ERA5Aggregated(out, self.geom, self.admin_level, "daily")
