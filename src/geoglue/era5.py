"""
ERA5 data download and data aggregation
---------------------------------------

This module downloads ERA5 daily level data and provides classes to
resample and aggregate data to administrative levels obtained from GADM
"""

from __future__ import annotations

import re
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import geopandas as gpd

from .types import CdoResampling
from .country import Country
from .memoryraster import MemoryRaster
from .zonal_stats import DatasetZonalStatistics
from .resample import resample


VAR_TO_STDNAMES: dict[str, str] = {"t2m": "2m_temperature", "tp": "total_precipitation"}
STDNAMES_TO_VAR: dict[str, str] = {v: k for k, v in VAR_TO_STDNAMES.items()}


class Dataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.metrics = data.metric.unique()
        self.metric = self.metrics[0] if len(self.metrics) == 1 else None
        if "daily_" in self.metrics[0]:
            self.temporal_scope = "daily"
            self.time = self.data.date.unique()
            years = self.data.date.dt.year
        elif "weekly_" in self.metrics[0]:
            self.temporal_scope = "weekly"
            self.time = self.data.isoweek.unique()
            years = self.data.isoweek.str[:4].map(int)
        else:
            raise ValueError("No temporal scope detected in dataset")
        min_year, max_year = int(years.min()), int(years.max())
        fmt_year = str(min_year) if min_year == max_year else f"{min_year}_{max_year}"
        self.weighted = "unweighted" not in self.metrics[0]
        self.admin_level = int(
            max(
                c.removeprefix("GID_")
                for c in self.data.columns
                if c.startswith("GID_")
            )
        )
        self.iso3 = self.data.ISO3.unique()[0]
        self.geometry = Country(self.iso3).admin(self.admin_level)
        self.filename = (
            f"{self.iso3.upper()}-{self.admin_level}-{fmt_year}-{self.metric}.parquet"
        )

    def __repr__(self):
        return repr(self.data)

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

    def weekly(self) -> Dataset:
        assert self.metric is not None
        self.data["isoweek"] = self.data.date.dt.strftime("%G-W%V")
        metric = self.metric.replace("daily_", "weekly_")
        agg = metric.split(".")[-1].removeprefix("weekly_")
        weekgroups = self.data.groupby(
            Country(self.iso3).list_admin_cols(self.admin_level) + ["isoweek"]
        )
        match agg:
            case "mean":
                df = weekgroups.value.mean()
            case "max":
                df = weekgroups.value.max()
            case "min":
                df = weekgroups.value.min()
            case "sum":
                df = weekgroups.value.sum()
        df = df.reset_index().sort_values("isoweek")  # type: ignore
        df["metric"] = metric
        df["ISO3"] = self.iso3
        return Dataset(df)

    def to_parquet(self, folder: Path = Path(".")):
        return self.data.to_parquet(folder / self.filename, index=False)


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
            cc.admin(level),
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
        operation: str = "mean",
        operation_params: str = "coverage_weight=area_spherical_km2",
        weighted: bool = True,
        min_date: datetime.date | None = None,
        max_date: datetime.date | None = None,
    ) -> pd.DataFrame:
        operation = f"{operation}({operation_params})"
        metric = variable if variable in STDNAMES_TO_VAR else VAR_TO_STDNAMES[variable]
        variable = (
            variable if variable in VAR_TO_STDNAMES else STDNAMES_TO_VAR[variable]
        )
        print(f"Zonal daily for {variable=} using {operation=}")
        const_cols = {"ISO3": self.iso3, "metric": f"era5.{metric}.{self.statistic}"}
        return super().zonal_stats(
            variable, operation, weighted, min_date, max_date, const_cols
        )
