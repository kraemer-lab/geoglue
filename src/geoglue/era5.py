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
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

from .types import CdoResampling
from .country import Country
from .memoryraster import MemoryRaster
from .zonal_stats import DatasetZonalStatistics
from .resample import resample


VAR_TO_STDNAMES: dict[str, str] = {"t2m": "2m_temperature"}
STDNAMES_TO_VAR: dict[str, str] = {v: k for k, v in VAR_TO_STDNAMES.items()}


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
        geom = Country(iso3).admin(admin_level)
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
            self.data.groupby(Country.list_admin_cols(self.admin_level) + ["isoweek"])
            .value.mean()
            .reset_index()
            .sort_values("isoweek")
        )
        df["metric"] = metric + ".weekly_mean"
        return ERA5Aggregated(df, self.geometry, self.admin_level, "weekly")


def infer_statistic(ds: xr.Dataset) -> str | None:
    "Infer temporal aggregation statistic from ERA5 NetCDF file"

    # daily_reduce functions by geoglue add a gg_statistic variable
    if "gg_statistic" in ds.attrs:
        return ds.attrs["gg_statistic"]
    last_line_history = ds.attrs["history"].split("\n")[-1]
    # Last line in the history attribute is of this form:
    # earthkit.transforms.aggregate.temporal.daily_reduce(2m_temperature_stream-oper, how=mean, **{'time_shift': {'hours': 7}, 'remove_partial_periods': True})
    # refers to the parameters passed to the daily_reduce function:
    # https://earthkit-transforms.readthedocs.io/en/latest/_api/transforms/aggregate/temporal/index.html#transforms.aggregate.temporal.daily_reduce
    if (m := re.match(r".*how=(\w+)", last_line_history)) and "daily_reduce" in m[0]:
        return "daily_" + m[1]
    return None


class ERA5ZonalStatistics(DatasetZonalStatistics):
    def __init__(
        self,
        filename: str,
        admin: str,
        weights: MemoryRaster | None = None,
        resampling: CdoResampling = CdoResampling.remapbil,
    ):
        iso3, level = admin.split("-")
        level = int(level)
        self.filename = filename
        self.iso3 = iso3.upper()
        self.admin_level = level
        cc = Country(iso3)
        self.resampling = resampling
        # perform resampling to weights grid if weights present
        if weights:
            resampled_file = resample(resampling, filename, weights)
        else:
            resampled_file = Path(filename)
        ds = xr.open_dataset(resampled_file)
        self.statistic = infer_statistic(ds)
        super().__init__(
            ds,
            cc.admin(level)
            weights,
            include_cols=cc.list_admin_cols(level),
            time_col="valid_time",
        )

    def __repr__(self):
        return f"""ERA5ZonalStatistics {self.iso3}-{self.admin_level} statistic={self.statistic}
        filename = {self.filename}
        variables = {self.variables}"""

    def zonal_daily(
        self,
        variable: str,
        operation: str = "mean(coverage_weight=area_spherical_m2)",
        weighted: bool = True,
        min_date: datetime.date | None = None,
        max_date: datetime.date | None = None,
    ) -> pd.DataFrame:
        metric = variable if variable in STDNAMES_TO_VAR else VAR_TO_STDNAMES[variable]
        variable = (
            variable if variable in VAR_TO_STDNAMES else STDNAMES_TO_VAR[variable]
        )
        const_cols = {"ISO3": self.iso3, "metric": f"era5.{metric}.{self.statistic}"}
        return super().zonal_stats(
            variable, operation, weighted, min_date, max_date, const_cols
        )
