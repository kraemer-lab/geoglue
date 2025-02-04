"""
ERA5 data download and data aggregation
---------------------------------------

This module downloads ERA5 daily level data and provides classes to
resample and aggregate data to administrative levels obtained from GADM
"""

from __future__ import annotations

import re
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

VARIABLE_MAPPINGS: dict[str, str] = {"2m_temperature": "t2m"}
INVERSE_VARIABLE_MAPPINGS: dict[str, str] = {v: k for k, v in VARIABLE_MAPPINGS.items()}
OMIT_VARIABLES = ["number", "latitude", "longitude", "valid_time"]


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
    weighted: bool = True

    @staticmethod
    def load(path: str | Path) -> ERA5Aggregated:
        path = Path(path)
        data = pd.read_parquet(path)

        # Standard filename nomenclature is of the form ISO3-admin_level-metric.parquet
        iso3 = data.attrs.get("iso3")
        if iso3 is None:
            iso3 = str(path.name).split("-")[0].upper()
        admin_level = int(data.attrs.get("admin_level", 0))
        if admin_level == 0:
            admin_level = int(str(path.name).split("-")[1])
        geom = GADM(iso3)[admin_level]
        metric = data.metric.unique().tolist()[0]
        temporal_scope = "weekly" if "weekly" in metric else "daily"
        weighted = data.attrs.get("weighted", True)
        return ERA5Aggregated(data, geom, admin_level, temporal_scope, weighted)

    def with_data(self, data: pd.DataFrame) -> ERA5Aggregated:
        return ERA5Aggregated(
            data, self.geometry, self.admin_level, self.temporal_scope, self.weighted
        )

    def save(self, path: str | Path | None = None) -> None:
        iso3 = (self.geometry.GID_0.unique().tolist()[0],)
        metric = self.data.metric.unique().tolist()[0]
        self.data.attrs = {
            "iso3": iso3,
            "admin_level": self.admin_level,
            "temporal_scope": self.temporal_scope,
            "weighted": self.weighted,
        }
        path = path or Path(f"{iso3}-{self.admin_level}-{metric}.parquet")
        self.data.to_parquet(path, index=False)

    def select(self, at: str):
        if self.temporal_scope == "weekly":
            return self.data[self.data.isoweek == at]
        else:
            at_date = datetime.date.fromisoformat(at)
            return self.data[self.data.date.dt.date == at_date]

    def select_values(self, at: str):
        data = self.select(at)
        return data.value.reset_index(drop=True)

    def plot(self, at: str):
        df = gpd.GeoDataFrame(self.select(at).merge(self.geometry))
        ax = df.plot("value", legend=True)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(df.metric.unique()[0])
        plt.show()

    def weekly(self) -> ERA5Aggregated:
        metric = self.data.metric.unique().tolist()[0]
        df = (
            self.data.groupby(GADM.list_admin_cols(self.admin_level) + ["isoweek"])
            .value.mean()
            .reset_index()
            .sort_values("isoweek")
        )
        df["metric"] = metric + ".weekly_mean"
        return ERA5Aggregated(df, self.geometry, self.admin_level, "weekly")


def infer_statistic(ds: xr.Dataset) -> str | None:
    "Infer temporal aggregation statistic from ERA5 NetCDF file"

    last_line_history = ds.attrs["history"].split("\n")[-1]
    # Last line in the history attribute is of this form:
    # earthkit.transforms.aggregate.temporal.daily_reduce(2m_temperature_stream-oper, how=mean, **{'time_shift': {'hours': 7}, 'remove_partial_periods': True})
    # refers to the parameters passed to the daily_reduce function:
    # https://earthkit-transforms.readthedocs.io/en/latest/_api/transforms/aggregate/temporal/index.html#transforms.aggregate.temporal.daily_reduce
    if (m := re.match(r".*how=(\w+)", last_line_history)) and "daily_reduce" in m[0]:
        return "daily_" + m[1]
    return None


class ERA5:
    def __init__(
        self,
        filename: str | Path,
        admin_level: tuple[str, int],
        population: MemoryRaster,
    ):
        self.filename = filename
        self.iso3, self.admin_level = admin_level
        self.geom = GADM(self.iso3)[self.admin_level]
        self.admin_cols = GADM.list_admin_cols(self.admin_level)
        self.dataset = xr.open_dataset(filename)

        # ERA5 stores longitude from the 0 to 360 scale
        # with 0 at the Greenwich meridian, counting east
        self.dataset.coords["longitude"] = (
            self.dataset.coords["longitude"] + 180
        ) % 360 - 180
        self.dataset = self.dataset.sortby(self.dataset.longitude)

        # Crop data to geometry extents
        extent_long, extent_lat = get_extents(self.geom)
        self.dataset = self.dataset.sel(longitude=extent_long, latitude=extent_lat)

        self.statistic = infer_statistic(self.dataset) or "unknown"
        self.population = population.mask(self.geom).astype(np.float32)
        self._variables: list[str] = [  # type: ignore
            INVERSE_VARIABLE_MAPPINGS.get(str(v), v)
            for v in self.dataset.variables
            if v not in OMIT_VARIABLES
        ]

    def __repr__(self):
        return (
            f"ERA5 {self.iso3} admin_level={self.admin_level} statistic={self.statistic}\n"
            f"filename  = {self.filename!r}\nvariables = {', '.join(self.variables)}"
        )

    @property
    def variables(self) -> list[str]:
        return self._variables

    def to_netcdf(self, *args, **kwargs) -> xr.Dataset:
        "Save NetCDF data"
        self.dataset.to_netcdf(*args, **kwargs)

    @cache
    def __getitem__(self, variable: str) -> xr.DataArray:
        varname = VARIABLE_MAPPINGS.get(variable, variable)
        return self.dataset[varname]

    def resample(
        self,
        variable: str,
        valid_time: int | datetime.date | str,
        resampling: Resampling = Resampling.bilinear,
    ) -> MemoryRaster:
        match valid_time:
            case str():
                arr = self[variable].sel(valid_time=valid_time)
            case datetime.date():
                arr = self[variable].sel(valid_time=valid_time.isoformat())
            case int():
                arr = self[variable].isel(valid_time=valid_time)
            case _:
                raise TypeError(f"{valid_time=} is of incorrect type")
        return MemoryRaster.from_xarray(arr).resample(self.population, resampling)

    def zonal_daily(
        self,
        variable: str,
        operation: str = "mean(coverage_weight=area_spherical_m2)",
        weighted: bool = True,
        min_date: datetime.date | None = None,
        max_date: datetime.date | None = None,
    ) -> ERA5Aggregated:
        da = self[variable]
        if weighted:
            operation = "weighted_" + operation
        min_date = min_date or da.valid_time.min().dt.date.item(0)
        max_date = max_date or da.valid_time.max().dt.date.item(0)
        assert max_date >= min_date, "End date must be later than start date"  # type: ignore

        exactextract_output_column = re.match(r"(\w+)(?=\()", operation).group(1)  # type: ignore
        # Empty dataframe with output columns
        out = pd.DataFrame(data=[], columns=self.admin_cols + ["value", "date"])

        for date in tqdm(pd.date_range(min_date, max_date, inclusive="both")):
            # NOTE: Rate-limiting step, as each weather 2D time slice needs to
            # be resampled to match the high resolution population grid
            rast = self.resample(variable, date)
            assert rast.shape == self.population.shape
            if weighted:
                df = rast.zonal_stats(
                    self.geom,
                    operation,
                    weights=self.population,
                    include_cols=self.admin_cols,
                ).rename(columns={exactextract_output_column: "value"})
            else:
                df = rast.zonal_stats(
                    self.geom,
                    operation,
                    include_cols=self.admin_cols,
                ).rename(columns={exactextract_output_column: "value"})

            df["date"] = date
            out = pd.concat([out, df])
        out["metric"] = f"era5.{variable}.{self.statistic}"
        out["isoweek"] = (
            out["date"].dt.isocalendar().year.astype(str)
            + "-W"
            + out["date"].dt.isocalendar().week.astype(str).str.zfill(2)
        )
        return ERA5Aggregated(out, self.geom, self.admin_level, "daily", weighted)
