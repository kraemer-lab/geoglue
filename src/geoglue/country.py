"""
This module contains the Country class that fetches geospatial data
(from GADM or geoBoundaries) for a particular country. It also supports
calculating extents or geospatial bounds, and calculating timezone offsets.
"""

import datetime
from functools import cache
from typing import Literal
from pathlib import Path

import pytz
import pycountry
import requests
import pandas as pd
import geopandas as gpd

import warnings

import geoglue

from .memoryraster import MemoryRaster
from .util import download_file
from .types import Bounds

GADM_VERSION = "4.1"

SHP_EXT = ["shp", "dbf", "prj", "shx"]  # shapefile extensions
GADM_EXT = SHP_EXT + ["cpg"]
GEOBOUNDARIES_EXT = SHP_EXT

# Maximum administrative level supported by each geospatial backend
SUPPORTED_BACKENDS = ["gadm", "geoboundaries"]
Backend = Literal["gadm", "geoboundaries"]
MAX_ADMIN: dict[Backend, int] = {"gadm": 3, "geoboundaries": 2}

# See significance of LOCALIZE_DATE in the Country constructor docstring
LOCALIZE_DATE = datetime.datetime(2022, 1, 1)


def geoboundaries_shapefile_url(url: str, adm: int) -> str:
    "Return shapefile URL for a particular admin level, geoboundaries backend"

    if (r := requests.get(f"{url}/ADM{adm}/")).status_code != 200:
        raise requests.ConnectionError(f"Error {r.status_code} when fetching {url=}")
    return r.json()["staticDownloadLink"]


def get_timezone_offset(tz: pytz.BaseTzInfo, localize_date: datetime.datetime) -> str:
    s = tz.localize(localize_date).strftime("%z")
    return s[:3] + ":" + s[3:]


class Country:
    def __init__(
        self,
        iso3: str,
        gadm_version: str = GADM_VERSION,
        crs: str = "EPSG:4326",
        backend: Backend = "gadm",
        localize_date: datetime.datetime = LOCALIZE_DATE,
        data_path: Path | None = None,
        timezone: pytz.BaseTzInfo | None = None,
        fetch_data: bool = True,
    ):
        """Constructs a Country class given an ISO3 code

        Parameters
        ----------
        iso3 : str
            ISO 3166-2 3-letter alpha code
        gadm_version : str
            If using GADM, GADM version to download, default=4.1
        crs : str
            Coordinate Reference System (CRS) to project downloaded geospatial data.
            Default is EPSG:4326, representing latitude and longitude
        backend : Backend
            Backend, must be one of `gadm` or `geoboundaries`
        localize_date : datetime.datetime
            Date where timezone is localised to, default=2022-01-01.

            Date used to localize the timezone obtained from pytz. Timezone
            names (such as Europe/Berlin) do not have a fixed offset due to daylight
            savings time changes, and the same timezone can have a different offset,
            usually in summer months. The exact date when DST starts also varies by year
            according to local policy shifts. We pick a specific date here to ensure that
            the localization is reproducible. The date is taken to be in the middle of
            winter in the Northern hemisphere when DST does not apply and the
            time offset follows standard time. For countries in the Southern hemisphere,
            the choice of this date may lead to non-standard (daylight savings) time
            being used.
        data_path : Path | None
            Data path to download data, by default ~/.local/share/geoglue
        timezone : pytz.BaseTzInfo | None
            Timezone, if not specified will be determined from country ISO3 code. The default
            mechanism works fine for countries with a unique timezone, but may not
            select the intended timezone for countries spanning multiple timezones.
        fetch_data : bool
            Whether to fetch data on initialisation, default=True. If set to False, data
            can be fetched later by calling fetch_shapefiles()
        """
        self.iso3 = iso3.upper()
        self.version = gadm_version if backend == "gadm" else None
        self.backend: Backend = backend
        self.iso2 = pycountry.countries.lookup(self.iso3).alpha_2
        self.timezones = pytz.country_timezones(self.iso2)  # type: ignore

        if timezone is None:
            if len(self.timezones) > 1:
                warnings.warn(
                    f"Multiple timezones for ISO3 country code {self.iso3} found {self.timezones}\n"
                    f"Selecting first timezone: {self.timezones[0]}"
                )
            self.timezone = pytz.timezone(self.timezones[0])  # type: ignore
        else:
            self.timezone = timezone

        self.timezone_offset = get_timezone_offset(self.timezone, localize_date)
        self.data_path = data_path or geoglue.data_path
        self._nodot_version = gadm_version.replace(".", "")
        self.crs = crs
        match backend:
            case "gadm":
                self.url = f"https://geodata.ucdavis.edu/gadm/gadm{gadm_version}/shp/gadm{self._nodot_version}_{iso3}_shp.zip"
                self.path_geodata = self.data_path / iso3 / f"gadm{self._nodot_version}"
                self.geodata_manifest = [
                    self.path_geodata
                    / f"gadm{self._nodot_version}_{self.iso3}_{i}.{ext}"
                    for i in [0, 1, 2]
                    for ext in GADM_EXT
                ]

            case "geoboundaries":
                self.url = f"https://www.geoboundaries.org/api/current/gbOpen/{iso3}/"
                self.path_geodata = self.data_path / iso3 / "geoboundaries"
                self.geodata_manifest = [
                    self.path_geodata / f"geoBoundaries-{self.iso3}-ADM{i}.{ext}"
                    for i in [0, 1, 2]
                    for ext in GEOBOUNDARIES_EXT
                ]
        self.path_population = self.data_path / iso3 / "worldpop"
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported geographic data {backend=}")

        self.path_geodata.mkdir(parents=True, exist_ok=True)
        self.path_population.mkdir(parents=True, exist_ok=True)

        if fetch_data and not self.geodata_manifest_ok:
            self.fetch_shapefiles()

    @property
    def geodata_manifest_ok(self) -> bool:
        "Returns True if all expected files in geodata manifest are present"
        return all(f.exists() for f in self.geodata_manifest)

    def fetch_shapefiles(self) -> bool:
        match self.backend:
            case "gadm":
                return download_file(
                    self.url, self.path_geodata / self.url.split("/")[-1]
                )
            case "geoboundaries":
                files = [
                    download_file(
                        geoboundaries_shapefile_url(self.url, adm=i), self.path_geodata
                    )
                    for i in range(3)
                ]
                return all(files)

    @property
    def bounds(self) -> Bounds:
        "Returns geographic bounds of the country"
        minx, miny, maxx, maxy = self.admin(0).total_bounds
        return Bounds(
            north=float(maxy), west=float(minx), south=float(miny), east=float(maxx)
        )

    @property
    def integer_bounds(self) -> Bounds:
        "Integral geographic bounds of the country"
        return self.bounds.integer_bounds()

    def population_raster(self, year: int) -> MemoryRaster:
        """Downloads and returns WorldPop population raster (1km resolution) for a particular year

        Data is only available from 2000-2020.

        Parameters
        ----------
        year : int
            Year for which to download data

        Returns
        -------
        MemoryRaster
            MemoryRaster representing the population data
        """
        if year < 2000 or year > 2020:
            raise ValueError(
                "Current population source 'worldpop' only has data from 2000-2020"
            )
        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/{year}/{self.iso3}/{self.iso3.lower()}_ppp_{year}_1km_Aggregated_UNadj.tif"
        output_path = self.path_population / url.split("/")[-1]
        if output_path.exists() or download_file(url, output_path):
            return MemoryRaster.read(output_path)
        else:
            raise requests.ConnectionError(f"Failed to download {url=}")

    def admin_cols(self, adm: int) -> list[str]:
        match self.backend:
            case "gadm":
                return [f"{c}_{i}" for c in ["GID", "NAME"] for i in range(1, adm + 1)]
            case "geoboundaries":
                return ["shapeID", "shapeName"]

    @cache
    def admin(self, adm: int) -> pd.DataFrame:
        "Returns administrative area dataframe with geometry for admin level `adm`, one of 1-3"
        if adm > MAX_ADMIN[self.backend]:
            raise ValueError(f"Unsupported {adm=} for backend={self.backend!r}")

        match self.backend:
            case "gadm":
                return gpd.read_file(
                    self.path_geodata
                    / f"gadm{self._nodot_version}_{self.iso3}_{adm}.shp"
                ).to_crs(self.crs)
            case "geoboundaries":
                return (
                    gpd.read_file(
                        self.path_geodata / f"geoBoundaries-{self.iso3}-ADM{adm}.shp"
                    )
                    .to_crs(self.crs)
                    .drop("shapeISO", axis=1)
                )
