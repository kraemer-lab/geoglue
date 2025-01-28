"""
ERA5 data download and data aggregation
---------------------------------------

This module downloads ERA5 daily level data and provides classes to
resample and aggregate data to administrative levels obtained from GADM
"""

from typing import Literal
from pathlib import Path
from functools import cache

import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

from memoryraster import MemoryRaster

# TODO: replace with GADM columns
ADM_COLS = ["ADM1_PCODE", "ADM1_EN", "ADM2_PCODE", "ADM2_EN"]

ERA5_VARIABLE_MAPPINGS = {"2m_temperature": "t2m"}


def get_extents(gdf: gpd.GeoDataFrame) -> tuple[slice, slice]:
    min_long, min_lat, max_long, max_lat = gdf.geometry.total_bounds
    return slice(int(min_long), int(max_long) + 1), slice(
        int(max_lat + 1), int(min_lat)
    )


class ERA5:
    def __init__(
        self,
        variables: list[str],
        year: int,
        country_iso3: str,
        admin_level: Literal[1, 2, 3],
        statistic: Literal["daily_mean", "daily_max", "daily_min"],
        filename: str | Path | None = None,
    ):
        self.statistic = statistic

    def download(self) -> bool:
        """Downloads ERA5 data

        Returns
        -------
        True if download successful, False otherwise
        """

    def set_population(self, raster: MemoryRaster):
        self._orig_population = raster

    @cache
    def era5_raster(self, variable: str) -> xr.DataArray:
        gdf = self.gadm_data
        varname = ERA5_VARIABLE_MAPPINGS.get(variable, variable)
        data = self.dataset[varname]

        # ERA5 stores longitude from the 0 to 360 scale
        # with 0 at the Greenwich meridian, counting east
        data.coords["longitude"] = (data.coords["longitude"] + 180) % 360 - 180
        data = data.sortby(data.longitude)
        # crop to geometry extent
        extent_long, extent_lat = get_extents(gdf)
        return data.sel(longitude=extent_long, latitude=extent_lat)

    def zonal_daily(
        self, variable: str, operation: str = "weighted_mean(coverage_weight=area_spherical_m2)"
    ) -> pd.DataFrame:
        da = self.era5_raster(variable)
        min_date = da.valid_time.min().dt.date.item(0)
        max_date = da.valid_time.max().dt.date.item(0)

        # Empty dataframe with output columns
        out = pd.DataFrame(data=[], columns=ADM_COLS + ["value", "date"])

        for date in tqdm(pd.date_range(min_date, max_date, inclusive="both")):
            arr = da.sel(valid_time=date.isoformat())
            rast = MemoryRaster.from_xarray(arr)
            df = rast.zonal_stats(
                self.gadm, operation, weights=self.population, include_cols=ADM_COLS
            ).rename(columns={"mean": "value"})
            df["date"] = date
            out = pd.concat([out, df])
        out["metric"] = f"era5.{variable}.{self.statistic}"
        out["isoweek"] = (
            out["date"].dt.isocalendar().year.astype(str)
            + "-W"
            + out["date"].dt.isocalendar().week.astype(str).str.zfill(2)
        )
        return out

    @staticmethod
    def weekly_mean(df: pd.DataFrame) -> pd.DataFrame:
        df = (
            df.groupby(ADM_COLS + ["isoweek"])
            .value.mean()
            .reset_index()
            .sort_values("isoweek")
        )
        df["metric"] += ".weekly_mean"
        return df

    @staticmethod
    def plot(df: pd.DataFrame, selector: str):
        if "isoweek" in df.columns:
            df = df[df.isoweek == selector]
        else:
            df = df[df.date == selector]
        return df.merge(self.gadm, ADM_COLS).plot()
