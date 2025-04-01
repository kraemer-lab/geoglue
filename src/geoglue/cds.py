"""
cdsapi helper utility module
----------------------------

This module uses ECMWF's ``cdsapi`` to downloads ERA5 hourly data and provides
utilities to time-shift the data to a particular timezone
"""

from __future__ import annotations

import re
import logging
import operator
import datetime
import zipfile
from pathlib import Path
from typing import NamedTuple, Literal, Iterable

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from .country import Country
from . import data_path


DAYS = [f"{i:02d}" for i in range(1, 32)]
MONTHS = [f"{i:02d}" for i in range(1, 13)]
TIMES = [f"{i:02d}:00" for i in range(24)]

ERA5_HOURLY_ACCUM_FILE = "data_stream-oper_stepType-accum.nc"
ERA5_HOURLY_INSTANT_FILE = "data_stream-oper_stepType-instant.nc"

CFGRIB_FILTER = {
    t: {
        "filter_by_keys": {"stream": ["oper"], "stepType": [t]},
        "encode_cf": ["parameter", "time", "geography", "vertical"],
        "time_dims": ["valid_time"],
    }
    for t in ["instant", "accum"]
}

Reducer = Literal["mean", "min", "max", "sum"]


def _is_hourly(ds: xr.Dataset, time_dim: str = "valid_time") -> bool:
    "Returns True if dataset is hourly"
    return sorted(set(ds[time_dim].dt.strftime("%H:%M").to_numpy())) == TIMES


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


def _daily_reduce(
    dataset: xr.Dataset, how: Reducer, dim: str = "valid_time"
) -> xr.Dataset:
    resampled = dataset.resample({dim: "D"})
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
        dim = None
        for coord_name, coord_data in self.instant.coords.items():
            if np.issubdtype(coord_data.dtype, np.datetime64):
                dim = coord_name
        if dim is None:
            raise ValueError("No time dimension found")
        if dim not in self.accum.coords:
            raise ValueError(f"Time dimension {dim=} not found in accum dataset")
        return dim

    def isel(self, **kwargs) -> CdsDataset:
        return CdsDataset(
            instant=self.instant.isel(**kwargs), accum=self.accum.isel(**kwargs)
        )

    def daily(self) -> CdsDataset:
        return CdsDataset(
            instant=_daily_reduce(self.instant, "mean"),
            accum=_daily_reduce(self.accum, "sum"),
        )

    def daily_max(self) -> xr.Dataset:
        return _daily_reduce(self.instant, "max")

    def daily_min(self) -> xr.Dataset:
        return _daily_reduce(self.instant, "min")

    def assign_coords(self, coords: dict) -> CdsDataset:
        return CdsDataset(
            instant=self.instant.assign_coords(coords),
            accum=self.accum.assign_coords(coords),
        )


class CdsPath(NamedTuple):
    instant: Path | None
    accum: Path | None

    def as_dataset(self) -> CdsDataset:
        return CdsDataset(
            instant=xr.open_dataset(self.instant), accum=xr.open_dataset(self.accum)
        )

    def exists(self) -> bool:
        assert self.instant or self.accum  # either must be present
        return (self.instant is None or self.instant.exists()) and (
            self.accum is None or self.accum.exists()
        )


def timeshift_hours(
    ds1: xr.Dataset | CdsDataset,
    ds2: xr.Dataset | CdsDataset,
    shift: int,
    dim: str = "valid_time",
) -> xr.Dataset | CdsDataset:
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
        ds1 = ds1.isel(**{dim: slice(-shift, None)})
        ds = (
            concat(ds1, ds2)
            if isinstance(ds1, CdsDataset)
            else xr.concat([ds1, ds2], dim=dim)
        )
        ds = ds.isel(**{dim: slice(None, -shift)})
    else:
        ds2 = ds2.isel(**{dim: slice(None, abs(shift))})
        ds = (
            concat(ds1, ds2)
            if isinstance(ds1, CdsDataset)
            else xr.concat([ds1, ds2], dim=dim)
        )
        ds = ds.isel(**{dim: slice(abs(shift), None)})

    time_shift = pd.Timedelta(hours=shift)
    if isinstance(ds, CdsDataset):
        time_coord = ds.instant.coords[dim] + time_shift  # type: ignore
    else:
        time_coord = ds.coords[dim] + time_shift
    time_coord = time_coord.assign_attrs({"time_shift": f"{shift}h"})
    return ds.assign_coords({dim: time_coord})


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
    ):
        self.iso3 = iso3.upper()
        self.geo_backend = geo_backend
        self.country = Country(iso3, backend=geo_backend)
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
            "area": list(self.country.integer_bounds),  # type: ignore
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
        hours = get_timezone_offset_hours(self.country.timezone_offset)
        if hours is None:
            raise ValueError(
                f"Can't perform timeshift for fractional timezone offset: {self.country.timezone_offset}"
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
        if len(set(p.parent for p in self.paths)) != 1:
            raise ValueError("All files in DatasetPool must be in same folder")
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
        if not all(self.path(year).as_dataset().is_hourly for year in self.years):
            raise ValueError("shift_hours option only supported for hourly DatasetPool")
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
                f"Positive shift_hours={self.shift_hours} require preceding year at {self.path(year-1)}"
            )
        if self.shift_hours < 0 and not self.path(year + 1).exists():
            raise FileNotFoundError(
                f"Negative shift_hours={self.shift_hours} require succeeding year at {self.path(year+1)}"
            )
        ds = self.path(year).as_dataset()
        time_dim = ds.get_time_dim()
        time_coord = ds.instant.coords[time_dim]
        ds = (
            timeshift_hours(
                self.path(year - 1).as_dataset(), ds, self.shift_hours, dim=time_dim
            )
            if self.shift_hours > 0
            else timeshift_hours(
                ds, self.path(year + 1).as_dataset(), self.shift_hours, dim=time_dim
            )
        )

        assert (ds.instant.coords[time_dim] == ds.accum.coords[time_dim]).all()
        if time_coord.min().values != np.datetime64(
            f"{year}-01-01"
        ) or time_coord.max().values != np.datetime64(f"{year}-12-31T23"):
            raise ValueError(
                "Improper alignment error: time dimension bounds do not match year bounds"
            )
        return ds
