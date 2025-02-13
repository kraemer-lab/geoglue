"Country information for a given ISO3 code, including geospatial data from GADM"

import datetime
from functools import cache

import pytz
import pycountry
import geopandas as gpd

import warnings

import geoglue
from .util import download_file

GADM_VERSION = "4.1"

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


def get_timezone_offset(tz: pytz.BaseTzInfo, localize_date: datetime.datetime) -> str:
    s = tz.localize(localize_date).strftime("%z")
    return s[:3] + ":" + s[3:]


class Country:
    def __init__(
        self,
        iso3: str,
        gadm_version: str = GADM_VERSION,
        crs: str = "EPSG:4326",
        localize_date: datetime.datetime = LOCALIZE_DATE,
        timezone: pytz.BaseTzInfo | None = None,
    ):
        self.iso3 = iso3.upper()
        self.version = gadm_version
        self.iso2 = pycountry.countries.lookup(self.iso3).alpha_2
        self.timezones = pytz.country_timezones(self.iso2)  # type: ignore

        if timezone is None:
            if len(self.timezones) > 1:
                warnings.warn(
                    "Multiple timezones for ISO3 country code {self.iso3} found {self.timezones}\n"
                    "Selecting first timezone: {self.timezones[0]}"
                )
            self.timezone = pytz.timezone(self.timezones[0])  # type: ignore
        else:
            self.timezone = timezone

        self.timezone_offset = get_timezone_offset(self.timezone, localize_date)
        self._nodot_version = gadm_version.replace(".", "")
        self.crs = crs
        self.url = f"https://geodata.ucdavis.edu/gadm/gadm{gadm_version}/shp/gadm{self._nodot_version}_{iso3}_shp.zip"
        self.gadm_path = geoglue.data_path / f"gadm{self._nodot_version}" / iso3
        if not self.gadm_path.exists():
            self.gadm_path.mkdir(parents=True)

    def fetch_shapefiles(self) -> bool:
        return download_file(self.url, self.gadm_path / self.url.split("/")[-1])

    @property
    def era5_extents(self) -> list[int]:
        minx, miny, maxx, maxy = self.admin(0).total_bounds
        return [int(maxy) + 1, int(minx), int(miny), int(maxx) + 1]

    @staticmethod
    def list_admin_cols(i) -> list[str]:
        cols = []
        for k in range(1, i + 1):
            cols.extend([it + "_" + str(k) for it in ["GID", "NAME"]])
        return cols

    @cache
    def admin(self, admin_level: int):
        if admin_level > 3:
            raise IndexError("Only admin level upto 3 supported in GADM")
        return gpd.read_file(
            self.gadm_path / f"gadm{self._nodot_version}_{self.iso3}_{admin_level}.shp"
        ).to_crs(self.crs)
