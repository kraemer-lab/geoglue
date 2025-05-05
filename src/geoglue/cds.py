"""
cdsapi helper utility module
----------------------------

This module uses ECMWF's ``cdsapi`` to downloads ERA5 hourly data and provides
utilities to time-shift the data to a particular timezone
"""

from __future__ import annotations

import re
import operator
import datetime
import warnings
import zipfile
from pathlib import Path
from typing import NamedTuple, Literal, Iterable

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from .country import Country
from .types import Bounds
from .util import find_unique_time_coord
from . import data_path


DAYS = [f"{i:02d}" for i in range(1, 32)]
MONTHS = [f"{i:02d}" for i in range(1, 13)]
TIMES = [f"{i:02d}:00" for i in range(24)]

ERA5_HOURLY_ACCUM_FILE = "data_stream-oper_stepType-accum.nc"
ERA5_HOURLY_INSTANT_FILE = "data_stream-oper_stepType-instant.nc"
DROP_VARS = ["number", "expver", "surface"]
CFGRIB_FILTER = {
    t: {
        "filter_by_keys": {"stream": ["oper"], "stepType": [t]},
        "encode_cf": ["parameter", "time", "geography", "vertical"],
        "time_dims": ["valid_time"],
    }
    for t in ["instant", "accum"]
}

INSTANT_AGG = ["mean", "min", "max"]
Reducer = Literal["mean", "min", "max", "sum"]


def _is_hourly(ds: xr.Dataset, time_dim: str = "valid_time") -> bool:
    "Returns True if dataset is hourly"
    return sorted(set(ds[time_dim].dt.strftime("%H:%M").to_numpy())) == TIMES


def get_first_monday(year: int) -> datetime.date:
    "Gets first Monday of the year"
    return datetime.datetime.strptime(f"{year}-W01-1", "%Y-W%W-%u").date()


def concat(a: CdsDataset, b: CdsDataset, time_dim: str = "valid_time") -> CdsDataset:
    instant_combined = xr.concat([a.instant, b.instant], dim=time_dim)
    accum_combined = xr.concat([a.accum, b.accum], dim=time_dim)
    return CdsDataset(instant=instant_combined, accum=accum_combined)


def get_timezone_offset_hours(offset: str) -> int | None:
    """Returns timezone offset in hours. Non-hourly offsets return None

    Parameters
    ----------
    offset
        String in the form [+-]HH:MM

    Examples
    --------
    >>> get_timezone_offset_hours("+05:00")
    5
    >>> get_timezone_offset_hours("-04:00")
    -4
    >>> get_timezone_offset_hours("+01:30")  # returns None

    Returns
    -------
    Timezone offset in hours, if fractional offset, then returns None
    """
    if offset[-3:] != ":00":  # fractional offset
        return None
    return int(offset.removesuffix(":00"))


def _time_reduce(
    dataset: xr.Dataset, freq: str, how: Reducer, dim: str = "valid_time", **kwargs
) -> xr.Dataset:
    resampled = dataset.resample({dim: freq}, **kwargs)
    assert how in ["min", "max", "mean", "sum"]
    return getattr(resampled, how)()


class CdsDataset(NamedTuple):
    instant: xr.Dataset
    accum: xr.Dataset

    def __eq__(self, other) -> bool:
        return self.instant.equals(other.instant) and self.accum.equals(other.accum)

    @property
    def is_hourly(self) -> bool:
        return _is_hourly(self.instant) and _is_hourly(self.accum)

    def get_time_dim(self) -> str:
        dim = find_unique_time_coord(self.instant)
        if dim not in self.accum.coords:
            raise ValueError(f"Time dimension {dim=} not found in accum dataset")
        return dim

    def isel(self, **kwargs) -> CdsDataset:
        return CdsDataset(
            instant=self.instant.isel(**kwargs), accum=self.accum.isel(**kwargs)
        )

    def daily(self) -> CdsDataset:
        return CdsDataset(
            instant=_time_reduce(self.instant, "D", "mean"),
            accum=_time_reduce(self.accum, "D", "sum"),
        )

    def daily_max(self) -> xr.Dataset:
        return _time_reduce(self.instant, "D", "max")

    def daily_min(self) -> xr.Dataset:
        return _time_reduce(self.instant, "D", "min")

    def assign_coords(self, coords: dict) -> CdsDataset:
        return CdsDataset(
            instant=self.instant.assign_coords(coords),
            accum=self.accum.assign_coords(coords),
        )


class CdsPath(NamedTuple):
    instant: Path | None
    accum: Path | None

    def as_dataset(self, drop_vars: list[str] = DROP_VARS) -> CdsDataset:
        instant = xr.open_dataset(self.instant)
        accum = xr.open_dataset(self.accum)
        to_drop_instant = set(instant.coords) & set(drop_vars)
        to_drop_accum = set(accum.coords) & set(drop_vars)
        return CdsDataset(
            instant=instant.drop_vars(to_drop_instant), accum=accum.drop_vars(to_drop_accum)
        )

    def exists(self) -> bool:
        assert self.instant or self.accum  # either must be present
        return (self.instant is None or self.instant.exists()) and (
            self.accum is None or self.accum.exists()
        )


def timeshift_hours(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    shift: int,
    dim: str = "valid_time",
    ) -> xr.Dataset:
    """Timeshift dataset by ``shift`` hours.

    If ``shift`` is a positive integer (longitude east), then that many hours
    are taken from the end of ds1 and attached onto ds2, with the end of ds2
    clipped to ensure that ds2 size remains the same.

    If ``shift`` is a negative integer (longitude west), then that many hours
    are taken from the beginning of ds2 and attached onto ds1, with the
    beginning of ds1 clipped to ensure that ds1 size remains the same.

    Checks are performed to ensure that ds1 and ds2 are contiguous in time,
    and that they are hourly data.

    Parameters
    ----------
    ds1
        First dataset, comprises most of the data in returned timeshifted
        dataset when shift < 0
    ds2
        Second dataset, comprises most of the data in returned timeshifted
        dataset when shift > 0
    shift
        Hours to timeshift, from [-12, 12], excluding 0.
    dim
        Name of the time dimension, optional

    Returns
    -------
    Timeshifted dataset

    Raises
    ------
    ValueError
        - Raised when no timeshift is performed when shift is zero
        - Raised when shift not in [-12, 12]
    """

    if shift == 0:
        raise ValueError(f"No timeshift required for {shift=}")
    if shift < -12 or shift > 12:
        raise ValueError(f"Timeshift valid for shift=-12..12, provided {shift=}")
    if shift > 0:
        ds1 = ds1.isel(**{dim: slice(-shift, None)})  # type: ignore
        ds = xr.concat([ds1, ds2], dim=dim)
        ds = ds.isel(**{dim: slice(None, -shift)})  # type: ignore
    else:
        ds2 = ds2.isel(**{dim: slice(None, abs(shift))})  # type: ignore
        ds = xr.concat([ds1, ds2], dim=dim)
        ds = ds.isel(**{dim: slice(abs(shift), None)})  # type: ignore

    time_shift = pd.Timedelta(hours=shift)
    time_coord = (ds.coords[dim] + time_shift).assign_attrs({"time_shift": f"{shift}h"})  # type: ignore
    return ds.assign_coords({dim: time_coord})


def timeshift_hours_cdsdataset(
    ds1: CdsDataset,
    ds2: CdsDataset,
    shift: int,
    dim: str = "valid_time",
) -> CdsDataset:
    """Timeshift CdsDataset by a integer number of hours

    This applies timeshift_hours() to the instant and accum parts of a
    CdsDataset. The main difference from applying timeshift_hours() directly is
    that we shift the time shift value for the accum dataset by -1. This is due
    to the fact that the accumulated and mean rate variables represent the hour
    to the time-stamp, that is, the data time-stamped as YYYY/MM/DD 00:00,
    represents the accumulation/mean-rate of the data for the time period 23:00
    to 00:00 for the date YYYY/MM/DD-1. See
    https://confluence.ecmwf.int/display/CKB/ERA5+family+post-processed+daily+statistics+documentation
    for context.

    Parameters
    ----------
    ds1
        First dataset, comprises most of the data in returned timeshifted
        dataset when shift <= 0
    ds2
        Second dataset, comprises most of the data in returned timeshifted
        dataset when shift > 0
    shift
        Hours to timeshift, from [-12, 12], excluding 0.
    dim
        Name of the time dimension, optional

    Returns
    -------
    Timeshifted dataset

    Raises
    ------
    ValueError
        - Raised when shift not in [-12, 12]
    """
    match shift:
        case 0:
            warnings.warn(
                "A zero timeshift will not change the 'instant' dataset, but will use ds2 to get the first hour for the 'accum' dataset"
            )
            return CdsDataset(
                instant=ds1.instant,
                accum=timeshift_hours(ds1.accum, ds2.accum, -1, dim),
            )
        case 1:
            return CdsDataset(
                instant=timeshift_hours(ds1.instant, ds2.instant, 1, dim),
                accum=ds2.accum,
            )
        case _:
            return CdsDataset(
                instant=timeshift_hours(ds1.instant, ds2.instant, shift, dim),
                accum=timeshift_hours(ds1.accum, ds2.accum, shift - 1, dim),
            )


def era5_extract_hourly_data(file: Path, extract_path: Path) -> CdsPath:
    "Extracts hourly data from downloaded zip file"
    if file.suffix != ".zip":
        raise ValueError(f"Not a valid zip {file=}")
    instant_file, accum_file = None, None
    with zipfile.ZipFile(file, "r") as zf:
        zf.extractall(extract_path / file.stem)
    if (accum_file := extract_path / file.stem / ERA5_HOURLY_ACCUM_FILE).exists():
        accum_file = accum_file.rename(extract_path / (file.stem + ".accum.nc"))
    if (instant_file := extract_path / file.stem / ERA5_HOURLY_INSTANT_FILE).exists():
        instant_file = instant_file.rename(extract_path / (file.stem + ".instant.nc"))
    if instant_file or accum_file:
        return CdsPath(instant=instant_file, accum=accum_file)
    else:
        raise ValueError(f"Error extracting hourly data from {file=}")


def grib_to_netcdf(file: Path, path: Path) -> CdsPath:
    assert file.suffix == ".grib"
    paths = {}
    for t in ["instant", "accum"]:
        ds = xr.open_dataset(file, engine="cfgrib", backend_kwargs=CFGRIB_FILTER[t])
        paths[t] = path / (file.stem + "." + t + ".nc")
        ds.to_netcdf(paths[t])
    return CdsPath(**paths)


class ReanalysisSingleLevels:
    def __init__(
        self,
        iso3: str,
        variables: list[str],
        geo_backend: Literal["gadm", "geoboundaries"] = "gadm",
        path: Path | None = None,
        stub: str = "era5",
        data_format: Literal["grib", "netcdf"] = "grib",
        bounds: Bounds | None = None,
        timezone_offset: str | None = None,
    ):
        self.iso3 = iso3.upper()
        self.geo_backend = geo_backend
        if (bounds is None) ^ (timezone_offset is None):
            raise ValueError(
                "Either none or both of bounds and timezone_offset must be specified"
            )
        if bounds and timezone_offset:
            self.bounds = bounds
            self.timezone_offset = timezone_offset
        else:
            # use country object to obtain bounds and timezone offset
            country = Country(iso3, backend=geo_backend)
            self.bounds = country.integer_bounds
            self.timezone_offset = country.timezone_offset
        self.variables = variables
        path = path or data_path / iso3 / "era5"
        if not path.exists():
            path.mkdir(parents=True)
        self.path = path
        self.stub = stub
        self.data_format: Literal["grib", "netcdf"] = data_format

    def __repr__(self):
        return (
            f"ReanalysisSingleLevels({self.iso3!r}, {self.variables!r}, geo_backend={self.geo_backend!r}, "
            f"path={self.path!r}, stub={self.stub!r}, data_format={self.data_format!r})"
        )

    def _cdsapi_request(self, year: int) -> dict[str, list[str] | list[int] | str]:
        "Returns cdsapi request dictionary for a given year"
        cur_year = datetime.datetime.now().year
        download_format = "unarchived" if self.data_format == "grib" else "zip"
        if year < 1940 or year > cur_year:
            raise ValueError(
                f"ERA5 reanalysis data only available from 1940-{cur_year}"
            )
        return {
            "product_type": ["reanalysis"],
            "variable": self.variables,
            "year": [str(year)],
            "month": MONTHS,
            "day": DAYS,
            "time": TIMES,
            "data_format": self.data_format,
            "download_format": download_format,
            "area": list(self.bounds),  # type: ignore
        }

    def get(self, year: int, skip_exists: bool = True) -> CdsPath | None:
        """Fetches hourly data for a particular year.

        An API key is needed for this function to work, see instructions at
        https://cds.climate.copernicus.eu/how-to-api

        Parameters
        ----------
        year
            Data is downloaded for this year
        skip_exists
            Skip downloading if zipfile or extracted contents exist, default True

        Returns
        -------
            Path of netCDF file that was written to disk
        """
        suffix = "grib" if self.data_format == "grib" else "zip"
        outfile = self.path / f"{self.iso3}-{year}-{self.stub}.{suffix}"
        accum_file = self.path / f"{self.iso3}-{year}-{self.stub}.accum.nc"
        instant_file = self.path / f"{self.iso3}-{year}-{self.stub}.instant.nc"
        if accum_file.exists() and instant_file.exists():
            return CdsPath(instant=instant_file, accum=accum_file)

        if not outfile.parent.exists():
            outfile.parent.mkdir(parents=True)
        if not skip_exists or not outfile.exists():
            client = cdsapi.Client()
            client.retrieve(
                "reanalysis-era5-single-levels", self._cdsapi_request(year), outfile
            )
        if outfile.exists():
            match self.data_format:
                case "grib":
                    return grib_to_netcdf(outfile, self.path)
                case "netcdf":
                    return era5_extract_hourly_data(outfile, self.path)

    def get_dataset_pool(self) -> DatasetPool:
        hours = get_timezone_offset_hours(self.timezone_offset)
        if hours is None:
            raise ValueError(
                f"Can't perform timeshift for fractional timezone offset: {self.timezone_offset}"
            )
        return DatasetPool(
            self.path.glob(f"{self.iso3}-????-{self.stub}.*.nc"), shift_hours=hours
        )


class DatasetPool:
    def __init__(self, paths: Iterable[Path], shift_hours: int = 0):
        """Instantiates a pool of yearly downloaded data that allows time-shifting"""
        self.paths = list(paths)

        regex = re.compile(r"^([A-Z]{3})-(\d{4})-(.*?)\.(instant|accum)\.nc$")
        # check that all files have the same stub
        matches = [regex.match(f.name) for f in self.paths]
        match_groups = [m.groups() for m in matches if m]
        parents = set(p.parent for p in self.paths)
        if len(parents) != 1:
            raise ValueError(f"All files in DatasetPool must be in same folder, found multiple parent folders: {parents}")
        self.folder = self.paths[0].parent
        iso3 = set(map(operator.itemgetter(0), match_groups))
        stubs = set(map(operator.itemgetter(2), match_groups))
        self.years = [
            int(x) for x in sorted(set(map(operator.itemgetter(1), match_groups)))
        ]
        if len(stubs) > 1 or len(iso3) > 1:
            raise ValueError(
                f"Multiple {iso3=} or {stubs=} not allowed in DatasetPool, specify a stricter path glob"
            )

        self.iso3 = iso3.pop()
        self.stub = stubs.pop()

        # all files should be hourly data
        for year in self.years:
            d = self.path(year).as_dataset()
            # TODO: Figure out why directly using 'if not self.path(year).as_dataset().is_hourly:'
            #       raises a segfault
            if not d.is_hourly:
                raise ValueError(
                    "shift_hours option only supported for hourly DatasetPool"
                )
        self.shift_hours = shift_hours

    def __repr__(self) -> str:
        return f"DatasetPool(shift_hours={self.shift_hours}, paths={self.paths!r}"

    def path(self, year: int) -> CdsPath:
        return CdsPath(
            instant=self.folder / f"{self.iso3}-{year}-{self.stub}.instant.nc",
            accum=self.folder / f"{self.iso3}-{year}-{self.stub}.accum.nc",
        )

    def __getitem__(self, year: int) -> CdsDataset:
        "Returns hourly dataset, time-shifted to local timezone"
        if year not in self.years:
            raise IndexError(
                f"{year=} not found in DatasetPool, valid years: {self.years}"
            )
        if self.shift_hours == 0:
            return self.path(year).as_dataset()
        if self.shift_hours > 0 and not self.path(year - 1).exists():
            raise FileNotFoundError(
                f"Positive shift_hours={self.shift_hours} require preceding year at {self.path(year - 1)}"
            )
        if self.shift_hours < 0 and not self.path(year + 1).exists():
            raise FileNotFoundError(
                f"Negative shift_hours={self.shift_hours} require succeeding year at {self.path(year + 1)}"
            )
        ds = self.path(year).as_dataset()
        time_dim = ds.get_time_dim()
        time_coord = ds.instant.coords[time_dim]
        if self.shift_hours > 0:
            ds = timeshift_hours_cdsdataset(
                self.path(year - 1).as_dataset(), ds, self.shift_hours, dim=time_dim
            )
        else:
            ds = timeshift_hours_cdsdataset(
                ds, self.path(year + 1).as_dataset(), self.shift_hours, dim=time_dim
            )
        assert (ds.instant.coords[time_dim] == ds.accum.coords[time_dim]).all()
        if time_coord.min().values != np.datetime64(
            f"{year}-01-01"
        ) or time_coord.max().values != np.datetime64(f"{year}-12-31T23"):
            raise ValueError(
                "Improper alignment error: time dimension bounds do not match year bounds"
            )
        return ds

    def weekly_reduce(
        self,
        year: int,
        vartype: Literal["instant", "accum"],
        how_daily: Reducer | None = None,
        how_weekly: Reducer | None = None,
        window: int = 0,
        time_dim: str = "valid_time",
    ) -> xr.Dataset:
        """Returns aggregated weekly dataset, time-shifted to local timezone.

        Dataset is aggregated to isoweeks, with week starting on Monday.

        Parameters
        ----------
        year
            Year to return weekly dataset for
        vartype
            One of `instant`, `accum` to select instantaneous or accumulative
            variables
        how_daily
            One of 'min', 'max', 'mean', default='mean'. Operation to aggregate
            from hourly to daily data. Ignored for accum vars, when we `sum` is used.
        how_weekly
            One of 'min', 'max', 'mean', default='mean'. Operation to aggregate
            from daily to weekly data. Ignored for accum vars, where `sum` is used.
        window
            Number of weeks to include before the first ISO week (first Monday
            of the year). This is useful when performing rolling operations
            which require `window` elements to be present to avoid NaNs.
        time_dim
            Time dimension to use, default='valid_time'

        Returns
        -------
        Dataset resampled to weekly frequency, with weeks starting on Monday (ISO weeks)
        """
        # Check correct reducer is picked
        match vartype:
            case "instant":
                how_daily = how_daily or "mean"
                how_weekly = how_weekly or "mean"
                if how_daily not in INSTANT_AGG or how_weekly not in INSTANT_AGG:
                    raise ValueError(
                        f"Invalid aggregation metric for 'instant' variable: {how_daily=} {how_weekly=} must be one of {INSTANT_AGG}"
                    )
            case "accum":
                how_daily = how_daily or "sum"
                how_weekly = how_weekly or "sum"
                if how_daily != "sum" or how_weekly != "sum":
                    raise ValueError(
                        "Invalid aggregation metric for 'accum' variable: must be 'sum' or unspecified"
                    )
        if not self.path(year - 1).exists() or not self.path(year + 1).exists():
            raise FileNotFoundError(
                f"Both data for {year - 1} and {year + 1} must be present for weekly statistics for {year=}"
            )

        match vartype:
            case "instant":
                ds = _time_reduce(self[year].instant, "D", how_daily)
                ds_prev = _time_reduce(self[year - 1].instant, "D", how_daily)
                ds_next = _time_reduce(self[year + 1].instant, "D", how_daily)

            case "accum":
                ds = _time_reduce(self[year].accum, "D", "sum")
                ds_prev = _time_reduce(self[year - 1].accum, "D", "sum")
                ds_next = _time_reduce(self[year + 1].accum, "D", "sum")

        if window > 0:  # needs previous year
            ds = xr.concat([ds_prev, ds, ds_next], dim=time_dim)
        else:
            ds = xr.concat([ds, ds_next], dim=time_dim)

        start_date = get_first_monday(year)
        end_date = get_first_monday(year + 1) - datetime.timedelta(days=1)
        if window > 0:
            start_date -= datetime.timedelta(days=7 * window)
        ds = ds.sel({time_dim: slice(start_date.isoformat(), end_date.isoformat())})

        return _time_reduce(ds, "W-MON", how_weekly, closed="left", label="left")
