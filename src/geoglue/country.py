"Country information for a given ISO3 code, including geospatial data from GADM"

import datetime
from functools import cache
from typing import Literal
from pathlib import Path

import pytz
import pycountry
import requests
import geopandas as gpd

import warnings

import geoglue

from .memoryraster import MemoryRaster
from .util import download_file

GADM_VERSION = "4.1"

# Maximum administrative level supported by each geospatial backend
SUPPORTED_BACKENDS = ["gadm", "geoboundaries"]
Backend = Literal["gadm", "geoboundaries"]
MAX_ADMIN: dict[Backend, int] = {"gadm": 3, "geoboundaries": 2}

# Date used to localize the timezone obtained from pytz. Timezone
# names (such as Europe/Berlin) do not have a fixed offset due to daylight
# savings time changes, and the same timezone can have a different offset,
# usually in summer months. The exact date when DST starts also varies by year
# according to local policy shifts. We pick a specific date here to ensure that
# the localization is reproducible. The date is taken to be in the middle of
# winter in the Northern hemisphere when DST does not apply and the
# time offset follows standard time. For countries in the Southern hemisphere,
# the choice of this date may lead to non-standard (daylight savings) time
# being used. Users can override the date used for localisation by passing the
# localize_date parameter to the Country constructor.
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
    ):
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
            case "geoboundaries":
                self.url = f"https://www.geoboundaries.org/api/current/gbOpen/{iso3}/"
                self.path_geodata = self.data_path / iso3 / "geoboundaries"
        self.path_population = self.data_path / iso3 / "worldpop"
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported geographic data {backend=}")

        self.path_geodata.mkdir(parents=True, exist_ok=True)
        self.path_population.mkdir(parents=True, exist_ok=True)

    def fetch_shapefiles(self) -> bool:
        match self.backend:
            case "gadm":
                return download_file(self.url, self.data_path / self.url.split("/")[-1])
            case "geoboundaries":
                files = [
                    download_file(
                        geoboundaries_shapefile_url(self.url, adm=i), self.data_path
                    )
                    for i in range(3)
                ]
                return all(files)

    @property
    def era5_extents(self) -> list[int]:
        minx, miny, maxx, maxy = self.admin(0).total_bounds
        return [int(maxy) + 1, int(minx), int(miny), int(maxx) + 1]

    def population_raster(self, year: int) -> MemoryRaster:
        if year < 2000 or year > 2020:
            raise ValueError(
                "Current population source 'worldpop' only has data from 2000-2020"
            )
        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/{year}/VNM/vnm_ppp_{year}_1km_Aggregated_UNadj.tif"
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
                return ["shapeID", "shapeName", "shapeISO"]

    @cache
    def admin(self, adm: int):
        if adm > MAX_ADMIN[self.backend]:
            raise ValueError(f"Unsupported {adm=} for backend={self.backend!r}")

        match self.backend:
            case "gadm":
                return gpd.read_file(
                    self.path_geodata / f"gadm{self._nodot_version}_{self.iso3}_{adm}.shp"
                ).to_crs(self.crs)
            case "geoboundaries":
                return gpd.read_file(
                    self.path_geodata / f"geoBoundaries-{self.iso3}-ADM{adm}.shp"
                )
