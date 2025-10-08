"""
This module uses ECMWF's ``cdsapi`` to downloads ERA5 hourly data and provides
utilities to time-shift the data to a particular timezone
"""

from __future__ import annotations

import json
import datetime
import logging
import operator
import re
import warnings
import zipfile
from pathlib import Path
from typing import Iterable, Literal, NamedTuple, Sequence

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
import geoglue

from .region import ZonedBaseRegion
from .util import find_unique_time_coord, get_first_monday

logger = logging.getLogger(__name__)

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


def is_end_of_month(d: datetime.date) -> bool:
    return (d + datetime.timedelta(days=1)).day == 1


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


def _time_reduce(
    dataset: xr.Dataset, freq: str, how: Reducer, dim: str = "valid_time", **kwargs
) -> xr.Dataset:
    resampled = dataset.resample({dim: freq}, **kwargs)
    assert how in ["min", "max", "mean", "sum"]
    return getattr(resampled, how)()


class CdsDataset(NamedTuple):
    """
    Tuple containing instant and accumulated variables from cdsapi
    """

    instant: xr.Dataset
    "Instant variables such as temperature and wind speed"

    accum: xr.Dataset
    "Accumulated variables, such as total precipitation and surface solar radiation"

    def __eq__(self, other) -> bool:
        return self.instant.equals(other.instant) and self.accum.equals(other.accum)

    @property
    def is_hourly(self) -> bool:
        "Returns whether dataset has hourly intervals"
        return _is_hourly(self.instant) and _is_hourly(self.accum)

    def get_time_dim(self) -> str:
        "Returns time dimension of the dataset"
        dim = find_unique_time_coord(self.instant)
        if dim not in self.accum.coords:
            raise ValueError(f"Time dimension {dim=} not found in accum dataset")
        return dim

    def isel(self, *args, **kwargs) -> CdsDataset:
        "Select slices by index from both instant and accumulated datasets"
        return CdsDataset(
            instant=self.instant.isel(*args, **kwargs),
            accum=self.accum.isel(*args, **kwargs),
        )

    def sel(self, *args, **kwargs) -> CdsDataset:
        "Select slices from both instant and accumulated datasets"
        return CdsDataset(
            instant=self.instant.sel(*args, **kwargs),
            accum=self.accum.sel(*args, **kwargs),
        )

    def equals(self, other: CdsDataset) -> bool:
        return self.instant.equals(other.instant) and self.accum.equals(other.accum)

    def daily(self) -> CdsDataset:
        "Returns CdsDataset corresponding to daily aggregation, mean for instant, sum for accumulated variables"
        return CdsDataset(
            instant=_time_reduce(self.instant, "D", "mean"),
            accum=_time_reduce(self.accum, "D", "sum"),
        )

    def daily_max(self) -> xr.Dataset:
        "Daily maximum of instant variable dataset"
        return _time_reduce(self.instant, "D", "max")

    def daily_min(self) -> xr.Dataset:
        "Daily minimum of instant variable dataset"
        return _time_reduce(self.instant, "D", "min")

    def assign_coords(self, coords: dict) -> CdsDataset:
        "Assigns coordinates to instant and accumulated variable datasets"
        return CdsDataset(
            instant=self.instant.assign_coords(coords),
            accum=self.accum.assign_coords(coords),
        )


class CdsPath(NamedTuple):
    """
    Tuple containing paths to instant and accumulated variables from cdsapi
    """

    instant: Path | None
    "Path to instant variable dataset"
    accum: Path | None
    "Path to accumulated variable dataset"

    def as_dataset(self, drop_vars: list[str] = DROP_VARS) -> CdsDataset:
        """
        Returns opened datasets for instant and accumulated variables

        Parameters
        ----------
        drop_vars
            Variables to drop, default=['number', 'expver', 'surface']

        Returns
        -------
        CdsDataset
            Dataset corresponding to CdsPath
        """
        instant = xr.open_dataset(self.instant)
        accum = xr.open_dataset(self.accum)
        to_drop_instant = set(instant.coords) & set(drop_vars)
        to_drop_accum = set(accum.coords) & set(drop_vars)
        return CdsDataset(
            instant=instant.drop_vars(to_drop_instant),
            accum=accum.drop_vars(to_drop_accum),
        )

    def exists(self) -> bool:
        "Returns True if dataset exists"
        assert self.instant or self.accum  # either must be present
        return (self.instant is None or self.instant.exists()) and (
            self.accum is None or self.accum.exists()
        )


def chunk_months(cds_data: CdsPath, stub: str, folder: Path) -> list[CdsPath]:
    ds = cds_data.as_dataset()
    min_year = pd.to_datetime(ds.instant.valid_time.min().values).year
    max_year = pd.to_datetime(ds.instant.valid_time.max().values).year
    # region name or iso3 is the first part by convention
    region_name = str(cds_data.instant.stem).split("-")[0]
    if min_year != max_year:
        raise ValueError(
            "chunk_months() is intended to be used for data within a calendar year"
        )
    year = min_year
    instant = ds.instant.sortby("valid_time")
    accum = ds.accum.sortby("valid_time")
    instant_files = []
    accum_files = []

    for month, m_instant in instant.groupby("valid_time.month"):
        assert is_end_of_month(pd.to_datetime(m_instant.valid_time.max().values).date())
        m_instant.to_netcdf(
            outpath := folder / f"{region_name}-{year}-{month:02d}-{stub}.instant.nc"
        )
        instant_files.append(outpath)

    for month, m_accum in accum.groupby("valid_time.month"):
        assert is_end_of_month(pd.to_datetime(m_accum.valid_time.max().values).date())
        m_accum.to_netcdf(
            outpath := folder / f"{region_name}-{year}-{month:02d}-{stub}.accum.nc"
        )
        accum_files.append(outpath)

    return [CdsPath(i, a) for i, a in zip(instant_files, accum_files)]


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
        Raised when shift not in [-12, 12]
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
    """Extracts hourly data from downloaded zip file

    Parameters
    ----------
    file
        zip file to open
    extract_path
        Path to extract to

    Returns
    -------
    CdsPath
        Path to extracted dataset
    """
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


def grib_to_netcdf(file: Path, path: Path | None = None) -> CdsPath:
    """Converts GRIB to netCDF

    Parameters
    ----------
    file
        GRIB file to open
    path
        Parent folder to save netCDF files, optional. If not specified
        write to the same folder as the GRIB file

    Returns
    -------
    CdsPath
        Path to converted netCDF dataset
    """
    path = path or file.parent
    assert file.suffix == ".grib"
    paths = {}
    for t in ["instant", "accum"]:
        ds = xr.open_dataset(file, engine="cfgrib", backend_kwargs=CFGRIB_FILTER[t])
        paths[t] = path / (file.stem + "." + t + ".nc")
        if paths[t].exists():
            paths[t].unlink()
        ds.to_netcdf(paths[t])
    return CdsPath(**paths)


def get_latest_era5_date() -> datetime.date:
    """Gets latest date when ERA5 data is available

    ERA5 releases data with a lag of 5 days"""
    return datetime.datetime.today().date() - datetime.timedelta(days=6)


class ReanalysisSingleLevels:
    """Fetch ERA5 reanalysis data from cdsapi for a particular country

    Parameters
    ----------
    region: Region
        Region for which to download data
    variables : list[str]
        List of variables to fetch
    path : Path | None
        Data path to download data to, optional. If not specified, downloads
        data to the default path, ``~/.local/share/geoglue``.
    stub : str
        Stub to use in filename, default=`era5`. This is used as part of the
        downloaded filename, e.g. ``VNM-2-2020-stub.accum.nc``
    data_format : Literal['grib', 'netcdf']
        Data format to download files in, one of `grib` or `netcdf`, default=`grib`.
        Downloading data in GRIB format allows downloading more variables. GRIB
        files are converted to netCDF, so both options result in identical data files.
    """

    def __init__(
        self,
        region: ZonedBaseRegion,
        variables: list[str],
        path: Path | None = None,
        stub: str = "era5",
        data_format: Literal["grib", "netcdf"] = "grib",
        admin_in_name: bool = False,
    ):
        self.region = region
        self.bbox = region.bbox.int()
        self.timezone_offset = region.tz
        self.variables = variables

        name = region.name.split(":")  # remove provider
        self.name = name[1] if len(name) > 1 else name[0]

        # Keep part before admin for name
        # This assumes that the boundaries for a particular name prefix
        # are same across administrative levels
        self.name_without_admin = self.name.split("-")[0]
        if not (
            path := path or geoglue.data_path / self.name_without_admin / "era5"
        ).exists():
            path.mkdir(parents=True)
        self.path = path
        self.stub = stub
        self.data_format: Literal["grib", "netcdf"] = data_format

    def __repr__(self):
        return (
            f"ReanalysisSingleLevels({self.region!r}, {self.variables!r}, "
            f"path={self.path!r}, stub={self.stub!r}, data_format={self.data_format!r})"
        )

    def _cdsapi_request(
        self,
        year: int,
        months: Sequence[int] | int | None = None,
        days: Sequence[int] | None = None,
        data_format: Literal["grib", "netcdf"] | None = None,
    ) -> dict[str, list[str] | list[int] | str]:
        "Returns cdsapi request dictionary for a given year, either full or partial"
        months = months or range(1, 13)
        days = days or range(1, 32)
        if isinstance(months, int):
            months = [months]
        if min(months) < 1 or max(months) > 12:
            raise ValueError(f"Invalid month supplied in {months}")
        if min(days) < 1 or max(days) > 31:
            raise ValueError(f"Invalid days supplied in {days}")

        data_format = data_format or self.data_format
        cur_year = datetime.datetime.today().year
        download_format = "unarchived" if data_format == "grib" else "zip"
        if year < 1940 or year > cur_year:
            raise ValueError(
                f"ERA5 reanalysis data only available from 1940-{cur_year}"
            )
        return {
            "product_type": ["reanalysis"],
            "variable": self.variables,
            "year": [str(year)],
            "month": [f"{i:02d}" for i in months],
            "day": [f"{i:02d}" for i in days],
            "time": TIMES,
            "data_format": self.data_format,
            "download_format": download_format,
            "area": self.bbox.to_list("cdsapi"),  # type: ignore
        }

    def get_current_year(
        self,
        start_date: datetime.date | str,
        end_date: datetime.date | str,
        skip_exists: bool = True,
    ) -> list[CdsPath] | None:
        """Fetches hourly data for a particular date range for the current year"""
        if isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.date.fromisoformat(end_date)
        chunks: list[CdsPath] = []
        cur = datetime.datetime.today().date()
        if start_date > end_date:
            raise ValueError(f"Improper date range: {start_date} -- {end_date}")
        if start_date.year != cur.year or end_date.year != cur.year:
            raise ValueError("""get_current_year() should only be used for dates in the current year
        for monthly downloads. To download an entire year at once, use get()""")
        if start_date > cur:
            raise ValueError(f"Date in future: {start_date}")
        if end_date > cur:
            raise ValueError(f"Date in future: {end_date}")

        # ERA5 releases data with a lag of 5 days, so offset by 6
        last_date_available = get_latest_era5_date()
        if end_date > last_date_available:
            warnings.warn(
                f"Last day ERA5 data is available for is {last_date_available}"
            )
            end_date = last_date_available
        # if end date falls on the last day of the previous month, partial data does not need to be downloaded
        partial_month = end_date.month == cur.month or (
            end_date.month == cur.month - 1
            and (end_date + datetime.timedelta(days=1)).day != 1
        )
        end_month = end_date.month

        # current month has to be downloaded separately
        if partial_month:
            end_month -= 1

        def _monthly_data(m: int) -> CdsPath:
            root = f"{self.name_without_admin}-{cur.year}-{m:02d}-{self.stub}"
            return CdsPath(
                self.path / f"{root}.instant.nc", self.path / f"{root}.accum.nc"
            )

        request_months = [
            m
            for m in range(start_date.month, end_month + 1)
            if not _monthly_data(m).exists()
        ]
        request_months.sort()
        month_str = ",".join(map(str, request_months))
        scratch_file = (
            geoglue.cache_path
            / f"{self.name_without_admin}-{cur.year}-{month_str}-era5.grib"
        )
        if request_months and (not scratch_file.exists() or not skip_exists):
            request = self._cdsapi_request(cur.year, request_months, data_format="grib")
            logger.info("Sending cdsapi request: %s", json.dumps(request))
            client = cdsapi.Client()
            client.retrieve("reanalysis-era5-single-levels", request, scratch_file)

            # Process downloaded file and chunk into monthly with the last chunk
            # being marked with _part. The last chunk will always be overwritten on
            # fresh request for the same month; whole months will not be redownloaded
            chunks.extend(
                chunk_months(grib_to_netcdf(scratch_file), self.stub, self.path)
            )

        # Always download current month to overwrite existing data
        partial_file = (
            self.path
            / f"{self.name_without_admin}-{cur.year}-{end_date.month:02d}_part-{self.stub}.grib"
        )
        if partial_month:
            request = self._cdsapi_request(
                cur.year, end_date.month, range(1, end_date.day + 1)
            )
            logger.info("Sending cdsapi request: %s", json.dumps(request))
            client = cdsapi.Client()
            client.retrieve("reanalysis-era5-single-levels", request, partial_file)
        chunks.append(grib_to_netcdf(partial_file))
        if not all(c.exists() for c in chunks):
            raise FileNotFoundError(
                "Chunks not found: " + str([c for c in chunks if not c.exists()]),
            )
        return chunks

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
        CdsPath
            Path of netCDF file that was written to disk
        """
        logger.info(
            "Get reanalysis data for region %s in %d for variables=%r",
            self.region.name,
            year,
            self.variables,
        )
        suffix = "grib" if self.data_format == "grib" else "zip"
        outfile = self.path / f"{self.name_without_admin}-{year}-{self.stub}.{suffix}"
        accum_file = (
            self.path / f"{self.name_without_admin}-{year}-{self.stub}.accum.nc"
        )
        instant_file = (
            self.path / f"{self.name_without_admin}-{year}-{self.stub}.instant.nc"
        )
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
        "Returns DatasetPool corresponding to downloaded data"
        hours = get_timezone_offset_hours(self.timezone_offset)
        if hours is None:
            raise ValueError(
                f"Can't perform timeshift for fractional timezone offset: {self.timezone_offset}"
            )
        return DatasetPool(
            self.path.glob(f"{self.name_without_admin}-????-*{self.stub}.*.nc"),
            shift_hours=hours,
        )


class DatasetPool:
    "Collection of ERA5 reanalysis data"

    def __init__(self, paths: Iterable[Path], shift_hours: int = 0):
        """Instantiates a pool of yearly downloaded data that allows time-shifting

        Parameters
        ----------
        paths : Iterable[Path]
            Paths that will be used to instantiate the dataset pool. All files in the DatasetPool
            must be in the same folder.
        shift_hours : int
            Integral number of hours to time shift the data, ranges from -12 to 12
        """
        self.paths = list(paths)

        regex = re.compile(r"^([A-Z]{3})-(\d{4})-(.*?)\.(instant|accum)\.nc$")
        part_regex = re.compile(
            r"^([A-Z]{3})-(\d{4}-0\d|1[0-2])(_part)?-(.*?)\.(instant|accum)\.nc$"
        )
        # check that all files have the same stub
        matches = [regex.match(f.name) for f in self.paths]
        part_matches = [part_regex.match(f.name) for f in self.paths]
        match_groups = [m.groups() for m in matches if m]
        part_match_groups = [m.groups() for m in part_matches if m]
        parents = set(p.parent for p in self.paths)
        if len(parents) != 1:
            raise ValueError(
                f"All files in DatasetPool must be in same folder, found multiple parent folders: {parents}"
            )
        self.folder = self.paths[0].parent
        iso3 = set(map(operator.itemgetter(0), match_groups))
        stubs = set(map(operator.itemgetter(2), match_groups))
        stubs = {x if "-" not in x else x.split("-")[-1] for x in stubs}
        self.years = [
            int(x) for x in sorted(set(map(operator.itemgetter(1), match_groups)))
        ]
        cur_year = int(datetime.datetime.today().year)
        if cur_year in self.years:
            self.years.remove(cur_year)
        self.part_chunks = sorted(set(map(operator.itemgetter(1), part_match_groups)))
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
        "Returns CdsDataset corresponding to a particular year"
        return CdsPath(
            instant=self.folder / f"{self.iso3}-{year}-{self.stub}.instant.nc",
            accum=self.folder / f"{self.iso3}-{year}-{self.stub}.accum.nc",
        )

    def get_current_year(
        self, start_date: datetime.date | str, end_date: datetime.date | str
    ) -> CdsDataset:
        latest = get_latest_era5_date()
        if isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.date.fromisoformat(end_date)

        cur = datetime.datetime.today().date()
        if not (cur.year == start_date.year == end_date.year):
            raise ValueError("get_current_year() only works for the current year")
        actual_start_date, actual_end_date = start_date, end_date

        # Shift start and end date to account for timeshift
        if self.shift_hours > 0:
            actual_start_date = start_date - datetime.timedelta(days=1)
        else:
            actual_end_date = end_date + datetime.timedelta(days=1)
        start_month, end_month = actual_start_date.month, actual_end_date.month
        required_months = [
            f"{m:02d}{'_part' if m == latest.month else ''}"
            for m in range(start_month, end_month + 1)
        ]

        def extract_month(f):
            mo = f.stem.split("-")[2]
            if mo.startswith("1") or mo.startswith("0"):
                return mo
            return None

        months = sorted(filter(None, set(map(extract_month, self.folder.glob("*.nc")))))
        existing_months = [
            m
            for m in months
            if m in required_months or m.replace("_part", "") in required_months
        ]
        if len(existing_months) != len(required_months):
            raise ValueError(
                "Required months do not exist, run ReanalysisSingleLevels.get_current_year() to fetch: "
                f"{set(required_months) - set(existing_months)}"
            )
        required_files = [
            CdsPath(
                self.folder / f"{self.iso3}-{cur.year}-{m}-{self.stub}.instant.nc",
                self.folder / f"{self.iso3}-{cur.year}-{m}-{self.stub}.accum.nc",
            )
            for m in existing_months
        ]
        ds = required_files.pop(0).as_dataset()
        time_dim = ds.get_time_dim()

        for r in required_files:
            ds = concat(ds, r.as_dataset(), time_dim)

        # ds0 corresponds to the time slice if there was no time shift
        ds0 = ds.sel({time_dim: slice(start_date.isoformat(), end_date.isoformat())})
        if self.shift_hours > 0:
            patch_slice = slice(
                actual_start_date.isoformat(), actual_start_date.isoformat() + "T23:00"
            )
            patch = ds.sel({time_dim: patch_slice})
            shifted_ds = timeshift_hours_cdsdataset(
                patch, ds0, self.shift_hours, time_dim
            )
            check_shifted_patch = shifted_ds.instant.isel(
                {time_dim: slice(0, self.shift_hours)}
            ).drop_vars(time_dim)
            patch_without_time = patch.instant.isel(
                {time_dim: slice(-self.shift_hours, None)}
            ).drop_vars(time_dim)
            assert check_shifted_patch.equals(patch_without_time)
        else:
            patch_slice = slice(
                actual_end_date.isoformat(), actual_end_date.isoformat() + "T23:00"
            )
            patch = ds.sel({time_dim: patch_slice})
            shifted_ds = timeshift_hours_cdsdataset(
                ds0, patch, self.shift_hours, time_dim
            )
            # shift_hours is negative so invert signs
            check_shifted_patch = shifted_ds.instant.isel(
                {time_dim: slice(self.shift_hours, None)}
            ).drop_vars(time_dim)
            patch_without_time = patch.instant.isel(
                {time_dim: slice(0, -self.shift_hours)}
            ).drop_vars(time_dim)
            assert check_shifted_patch.equals(patch_without_time)

        return shifted_ds

    def __getitem__(self, year: int) -> CdsDataset:
        "Returns hourly dataset for a particular year, time-shifted to local timezone"
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
        xr.Dataset
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
