"""
This module contains the Region class that has functions to fetch geospatial data
(from GADM or geoBoundaries) for a particular country, as well as structures
to make work with arbitrary shapefiles easier. It also supports
calculating extents or geospatial bounds, and calculating timezone offsets.
"""

import logging
import datetime
from functools import cache
from typing import TypedDict
from pathlib import Path

import pytz
import pycountry
import requests
import geopandas as gpd

import geoglue

from .memoryraster import MemoryRaster
from .util import download_file
from .types import Bbox

logger = logging.getLogger(__name__)

SHP_EXT = ["shp", "dbf", "prj", "shx"]  # shapefile extensions
GADM_EXT = SHP_EXT + ["cpg"]
GEOBOUNDARIES_EXT = SHP_EXT

# See significance of LOCALIZE_DATE in the get_timezone()
LOCALIZE_DATE = datetime.datetime(2022, 1, 1)


class Region(TypedDict):
    "Dictionary representing a geospatial region"

    name: str
    "Human readable name"

    path: str | Path
    "Path to shapefile"

    pk: str
    "Column ID that is used as primary key to identify regions in shapefile"

    tz: str
    "Timezone as [+-]HH:MM from UTC"

    url: str
    "URL from which data was downloaded"

    bbox: Bbox
    "Geospatial bounding box"


def read_region(r: Region) -> gpd.GeoDataFrame:
    "Reads a region shapefile"
    df = gpd.read_file(r["path"])
    # drop shapeISO column which is just None
    if "shapeISO" in df.columns:
        return df.drop(columns=["shapeISO"])  # type: ignore
    return df


def get_timezone(iso3: str, localize_date: datetime.datetime) -> str | None:
    """Returns unique timezone offset for a country with ISO3 code

    Parameters
    ----------
    iso3 : str
        ISO3 code of country
    localize_date : datetime.datetime
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

    Returns
    -------
    str | None
        Timezone offset as [+-]HH:MM from UTC if unique timezone found, None otherwise
    """
    iso2 = pycountry.countries.lookup(iso3).alpha_2
    if len(timezones := pytz.country_timezones[iso2]) > 1:
        logging.info(
            "No unique timezone for %s spanning multiple zones: %r, returning None",
            iso3,
            timezones,
        )
        return None
    tz = pytz.timezone(timezones[0])
    tz_str = tz.localize(localize_date).strftime("%z")
    return tz_str[:3] + ":" + tz_str[3:]


def _geoboundaries_shapefile_url(url: str, adm: int) -> str:
    "Return shapefile URL for a particular admin level, geoboundaries backend"

    if (r := requests.get(f"{url}/ADM{adm}/")).status_code != 200:
        raise requests.ConnectionError(f"Error {r.status_code} when fetching {url=}")
    return r.json()["staticDownloadLink"]


def get_bbox(path: str | Path) -> Bbox:
    "Gets bounding box of a shapefile"
    data = gpd.read_file(path)
    return Bbox(*data.total_bounds)


def gadm(
    iso3: str,
    admin: int,
    localize_date: datetime.datetime = LOCALIZE_DATE,
    data_path: Path | None = None,
    tzoffset: str | None = None,
) -> Region:
    """
    Returns GADM Region data

    Parameters
    ----------
    iso3 : str
        Country ISO3 code
    admin : int
        Admin level, one of 1, 2 or 3
    localize_date : datetime.datetime
        Date where timezone is localised to, default=2022-01-01.
        See :meth:`get_timezone()` for information about this parameter
    data_path : Path | None
        Optional. If specified, sets the data path where shapefiles will be downloaded,
        otherwise defaults to ``~/.local/share/geoglue``
    tzoffset : str | None
        Optional, specifies timezone offset as [+-]HH:MM from UTC. If not specified
        is automatically inferred from country ISO3 code. Auto-detection is only
        performed for countries with one timezone, and this parameter is mandatory
        for countries spanning multiple timezones.

    Returns
    -------
    Region
        Region data representing GADM information for a country at a
        particular admin level
    """
    if admin not in [1, 2, 3]:
        raise ValueError("Unsupported administrative level, must be one of 1-3")
    data_path = data_path or geoglue.data_path
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip"
    path_geodata = data_path / iso3 / "gadm41"
    path_geodata.mkdir(parents=True, exist_ok=True)
    manifest = [
        path_geodata / f"gadm41_{iso3}_{i}.{ext}"
        for i in [0, 1, 2, 3]
        for ext in GADM_EXT
    ]
    if not all(f.exists() for f in manifest):
        logging.info("Missing GADM data for %s, downloading data", iso3)
        if not download_file(url, path_geodata / url.split("/")[-1]):
            raise ConnectionError(
                "GADM data download failed %s -> %s", url, path_geodata
            )
        logging.info("GADM data downloaded to %s", path_geodata)
    path = path_geodata / f"gadm41_{iso3}_{admin}.shp"
    if (tzoffset := tzoffset or get_timezone(iso3, localize_date)) is None:
        raise ValueError("No unique timezone offset found or supplied")
    return {
        "name": f"{iso3}-{admin}",
        "path": path,
        "pk": f"GID_{admin}",
        "tz": tzoffset,
        "url": url,
        "bbox": get_bbox(path),
    }


def geoboundaries(
    iso3: str,
    admin: int,
    localize_date: datetime.datetime = LOCALIZE_DATE,
    data_path: Path | None = None,
    tzoffset: str | None = None,
) -> Region:
    """
    Returns geoBoundaries Region data

    Parameters
    ----------
    iso3 : str
        Country ISO3 code
    admin : int
        Admin level, one of 1, 2
    localize_date : datetime.datetime
        Date where timezone is localised to, default=2022-01-01.
        See :meth:`get_timezone()` for information about this parameter
    data_path : Path | None
        Optional. If specified, sets the data path where shapefiles will be downloaded,
        otherwise defaults to ``~/.local/share/geoglue``
    tzoffset : str | None
        Optional, specifies timezone offset as [+-]HH:MM from UTC. If not specified
        is automatically inferred from country ISO3 code. Auto-detection is only
        performed for countries with one timezone, and this parameter is mandatory
        for countries spanning multiple timezones.

    Returns
    -------
    Region
        Region data representing geoBoundaries information for a country at a
        particular admin level
    """
    if admin not in [1, 2]:
        raise ValueError("Unsupported administrative level, must be one of 1-2")
    data_path = data_path or geoglue.data_path
    url = f"https://www.geoboundaries.org/api/current/gbOpen/{iso3}/"
    path_geodata = data_path / iso3 / "geoboundaries"
    path_geodata.mkdir(parents=True, exist_ok=True)
    manifest = [
        path_geodata / f"geoboundaries_{iso3}_ADM{admin}.{ext}"
        for ext in GEOBOUNDARIES_EXT
    ]
    if not all(f.exists() for f in manifest):
        logging.info("Missing geoBoundaries data for %s, downloading data", iso3)
        if not download_file(
            _geoboundaries_shapefile_url(url, adm=admin), path_geodata
        ):
            raise ConnectionError(
                "geoBoundaries data download failed %s -> %s",
                _geoboundaries_shapefile_url(url, adm=admin),
            )
        logging.info("geoBoundaries data downloaded to %s", path_geodata)
    path = path_geodata / f"geoBoundaries-{iso3}-ADM{admin}.shp"
    if (tzoffset := tzoffset or get_timezone(iso3, localize_date)) is None:
        raise ValueError("No unique timezone offset found or supplied")
    return {
        "name": f"{iso3}-{admin}",
        "path": path,
        "pk": "shapeID",
        "tz": tzoffset,
        "url": url,
        "bbox": get_bbox(path),
    }


@cache
def get_worldpop_1km(
    iso3: str, year: int, data_path: Path | None = None
) -> MemoryRaster:
    """Downloads and returns WorldPop population raster (1km resolution) for a particular year

    Data is only available from 2000-2020.

    Parameters
    ----------
    iso3 : str
        ISO3 code of country
    year : int
        Year for which to download data
    data_path : Path | None
        Optional. If specified, sets the data path where population data will be downloaded,
        otherwise defaults to ``~/.local/share/geoglue``

    Returns
    -------
    MemoryRaster
        MemoryRaster representing the population data
    """
    if year < 2000 or year > 2020:
        raise ValueError("Worldpop population data is only available from 2000-2020")
    path_population = (data_path or geoglue.data_path) / iso3 / "worldpop"
    path_population.mkdir(parents=True, exist_ok=True)
    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/{year}/{iso3}/{iso3.lower()}_ppp_{year}_1km_Aggregated_UNadj.tif"
    output_path = path_population / url.split("/")[-1]
    if output_path.exists() or download_file(url, output_path):
        return MemoryRaster.read(output_path)
    else:
        raise requests.ConnectionError(f"Failed to download {url=}")
