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
from typing import Literal
from pathlib import Path
from functools import cache
from dataclasses import dataclass

import cdo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

from .types import CdoGriddes, CdoResampling
from .gadm import GADM
from .memoryraster import MemoryRaster

VAR_TO_STDNAMES: dict[str, str] = {"t2m": "air_temperature"}
STDNAMES_TO_VAR: dict[str, str] = {v: k for k, v in VAR_TO_STDNAMES.items()}


_cdo = cdo.Cdo()


def get_extents(gdf: gpd.GeoDataFrame) -> tuple[slice, slice]:
    min_long, min_lat, max_long, max_lat = gdf.geometry.total_bounds
    return slice(int(min_long), int(max_long) + 1), slice(
        int(max_lat) + 1, int(min_lat)
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


def resample(
    resampling: CdoResampling,
    infile: str | Path,
    target: MemoryRaster,
    outfile: str | Path | None = None,
    skip_exists=True,
) -> Path:
    "Resamples input file to output file using CDO's resampling to a target raster grid"
    if isinstance(infile, str):
        infile = Path(infile)
    if outfile is None:
        outfile = infile.parent / f"{infile.stem}_{resampling.name}.nc"
    if Path(outfile).exists() and skip_exists:
        return Path(outfile)
    with tempfile.NamedTemporaryFile(suffix=".txt") as griddes:
        Path(griddes.name).write_text(str(target.griddes))

        match resampling:
            case CdoResampling.remapbil:
                _cdo.remapbil(griddes.name, input=str(infile), output=str(outfile))
            case CdoResampling.remapdis:
                _cdo.remapdis(griddes.name, input=str(infile), output=str(outfile))
        return Path(outfile)


def geom_plot(df: pd.DataFrame, geometry: gpd.GeoDataFrame, col: str = "value"):
    return gpd.GeoDataFrame(df.merge(geometry)).plot(col)


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
        self.weighted = not (weights is None)
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
        operation: str = "mean(coverage_weight=area_spherical_m2)",
        weighted: bool = True,
        min_date: datetime.date | None = None,
        max_date: datetime.date | None = None,
        const_cols: dict[str, str] | None = None,
    ) -> ERA5Aggregated:
        da = self[variable]
        # shape after dropping time axis should be identical
        if weighted:
            if self.weights is None:
                raise ValueError(
                    "Weighted zonal statistics requested but no weights supplied"
                )
            assert (
                da.shape[1:] == self.weights.shape
            ), f"Variable shape {da.shape[1:]} and weights shape {self.weights.shape} must be identical"
            operation = "weighted_" + operation
        min_date = min_date or da[self.time_col].min().dt.date.item(0)
        max_date = max_date or da[self.time_col].max().dt.date.item(0)
        assert max_date >= min_date, "End date must be later than start date"  # type: ignore

        exactextract_output_column = re.match(r"(\w+)(?=\()", operation).group(1)  # type: ignore
        # Empty dataframe with output columns
        out = pd.DataFrame(data=[], columns=self.include_cols + ["value", "date"])

        for date in tqdm(pd.date_range(min_date, max_date, inclusive="both")):
            rast = MemoryRaster.from_xarray(da.sel({self.time_col: date}))
            if weighted:
                df = rast.zonal_stats(
                    self.geom,
                    operation,
                    weights=self.weights,
                    include_cols=self.include_cols,
                ).rename(columns={exactextract_output_column: "value"})
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
        gadm = GADM(iso3)
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
            gadm[level],
            weights,
            include_cols=gadm.list_admin_cols(level),
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
