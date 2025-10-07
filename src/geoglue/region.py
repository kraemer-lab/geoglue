"""
This module contains the Region class that has functions to fetch geospatial data
(from GADM or geoBoundaries) for a particular country, as well as structures
to make work with arbitrary shapefiles easier. It also supports
calculating extents or geospatial bounds, and calculating timezone offsets.
"""

from __future__ import annotations

import re
import logging
import datetime
from typing import NamedTuple, Mapping, Literal
from pathlib import Path

import pytz
import pycountry
import requests
import geopandas as gpd

import tomli as toml
import geoglue

from .util import download_file
from .types import Bbox

logger = logging.getLogger(__name__)

SHP_EXT = ["shp", "dbf", "prj", "shx"]  # shapefile extensions
GADM_EXT = SHP_EXT + ["cpg"]
GEOBOUNDARIES_EXT = SHP_EXT

# See significance of LOCALIZE_DATE in the get_timezone()
LOCALIZE_DATE = datetime.datetime(2022, 1, 1)


class Region(NamedTuple):
    "Tuple representing a geospatial region"

    name: str
    "Region identifier without spaces"

    admin_files: Mapping[int, str | Path]
    "Path to shapefiles, indexed by administrative level"

    pk: dict[int, str] | str
    """Column ID that is used as primary key to identify regions
    in shapefile, indexed by administrative level.

    If str, is the same for every administrative level
    """

    tz: str
    """Timezone offset from UTC.

    Expressed as [+-]HH:MM, e.g. +01:00 for CET timezone
    """

    url: str
    "URL from which data was downloaded"

    bbox: Bbox
    "Geospatial bounding box"

    admin: int | None = None
    "If specified, refers to a specific administrative level"

    def __str__(self) -> str:
        return " ".join(
            [
                self.name,
                str(self.bbox),
                self.tz,
                self.url,
            ]
        )

    def read_admin(self, admin: int | None = None) -> gpd.GeoDataFrame:
        "Reads a region shapefile"
        admin = admin or self.admin
        if admin is None:
            raise ValueError(
                "Administrative level not specified, and no Region.admin set"
            )
        if admin not in self.admin_files:
            raise KeyError(
                f"Administrative level {admin} shapefile not defined for {self.name!r}"
            )
        df = gpd.read_file(self.admin_files[admin])
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
        logger.info(
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
    localize_date: datetime.datetime = LOCALIZE_DATE,
    data_path: Path | None = None,
    tzoffset: str | None = None,
    admin: int | None = None,
) -> Region:
    """
    Returns GADM Region data

    Parameters
    ----------
    iso3 : str
        Country ISO3 code
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
    admin : int | None
        Optional, sets administrative level in returned Region object

    Returns
    -------
    Region
        Region data representing GADM information for a country at a
        particular admin level
    """
    iso3 = iso3.upper()
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
        logger.info("Missing GADM data for %s, downloading data", iso3)
        if not download_file(url, path_geodata / url.split("/")[-1]):
            raise ConnectionError(
                "GADM data download failed %s -> %s", url, path_geodata
            )
        logger.info("GADM data downloaded to %s", path_geodata)
    admins = {
        int(path.stem.split("_")[-1]): path for path in path_geodata.glob("*.shp")
    }
    if (tzoffset := tzoffset or get_timezone(iso3, localize_date)) is None:
        raise ValueError("No unique timezone offset found or supplied")
    return Region(
        f"gadm:{iso3}",
        admins,
        {i: f"GID_{i}" for i in admins},
        tzoffset,
        url,
        get_bbox(admins[1]),
        admin,
    )


def geoboundaries(
    iso3: str,
    localize_date: datetime.datetime = LOCALIZE_DATE,
    data_path: Path | None = None,
    tzoffset: str | None = None,
    admin: int | None = None,
) -> Region:
    """
    Returns geoBoundaries Region data

    Parameters
    ----------
    iso3 : str
        Country ISO3 code
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
    admin : int | None
        Optional, sets administrative level in returned Region object

    Returns
    -------
    Region
        Region data representing geoBoundaries information for a country at a
        particular admin level
    """
    iso3 = iso3.upper()
    data_path = data_path or geoglue.data_path
    url = f"https://www.geoboundaries.org/api/current/gbOpen/{iso3}/"
    path_geodata = data_path / iso3 / "geoboundaries"
    path_geodata.mkdir(parents=True, exist_ok=True)
    manifest = [path_geodata / f"geoboundaries-{iso3}-ADM{adm}.shp" for adm in [1, 2]]
    for f in manifest:
        if f.exists():
            continue
        adm = int(f.stem.split("-")[-1][3:])
        logger.info("Missing geoBoundaries data for %s, downloading data", iso3)
        if not download_file(_geoboundaries_shapefile_url(url, adm=adm), path_geodata):
            raise ConnectionError(
                "geoBoundaries data download failed %s -> %s",
                _geoboundaries_shapefile_url(url, adm=adm),
            )
        logger.info("geoBoundaries data downloaded to %s", path_geodata)
    admins = {i: path_geodata / f"geoBoundaries-{iso3}-ADM{i}.shp" for i in [1, 2]}
    if (tzoffset := tzoffset or get_timezone(iso3, localize_date)) is None:
        raise ValueError("No unique timezone offset found or supplied")
    return Region(
        f"gb:{iso3}", admins, "shapeID", tzoffset, url, get_bbox(admins[1]), admin
    )


def get_region(
    name: str,
    file: str | Path | None = None,
    fallback: Literal["gadm", "geoboundaries"] = "gadm",
    **kwargs,
) -> Region:
    """Returns region from file or fallback to GADM or geoBoundaries

    Parameters
    ----------
    name : str
        Name of the region, e.g. 'VNM', 'HCMC'
    file : str | Path | None
        TOML file from which regions should be read. If not specified,
        fallback to GADM or geoBoundaries
    fallback : Literal["gadm", "geoboundaries"]
        Default fallback provider, used when file is not specified or
        region name not found in the TOML file
    **kwargs
        Extra parameters passed to :meth:`gadm` or :meth:`geoboundaries`

    Returns
    -------
    Region
    """
    if "-" in name:
        name, admin = name.split("-")
        admin = int(admin)
    else:
        admin = None
    if file is not None and Path(file).exists():
        data = toml.loads(Path(file).read_text())
    else:
        data = {}
    if name not in data:
        # fallback to gadm or geoboundaries
        match fallback:
            case "gadm":
                return gadm(name, **kwargs)
            case "geoboundaries":
                return geoboundaries(name, **kwargs)
    region_dict = data[name]
    admin_files = {int(k): Path(v) for k, v in region_dict["admin_files"].items()}
    if isinstance(region_dict["pk"], dict):
        pk = {int(k): v for k, v in region_dict["pk"].items()}
    else:
        pk = region_dict["pk"]

    tz = region_dict["tz"]
    if not re.match(r"[+-][01]\d:([03]0|45)", tz):
        raise ValueError(f"Invalid timezone in region {name}: {tz}")
    if not (url := region_dict["url"]).startswith("https://"):
        raise ValueError(f"Invalid URL in region {name}: {url}")
    minx, miny, maxx, maxy = region_dict["bbox"]
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise ValueError(
            f"Invalid bounds for region {name}: {minx},{miny},{maxx},{maxy}"
        )
    bbox = Bbox(minx, miny, maxx, maxy)
    if admin is not None and admin not in admin_files:
        raise ValueError(f"No shapefile specified for {admin=}, which is required")
    return Region(name, admin_files, pk, tz, url, bbox, admin)
