"GADM data fetcher"

from functools import cache

import geopandas as gpd

import geoglue
from .util import download_file

GADM_VERSION = "4.1"


class GADM:
    def __init__(self, iso3: str, version: str = GADM_VERSION, crs: str = "EPSG:4326"):
        self.iso3 = iso3.upper()
        self.version = version
        self._nodot_version = version.replace(".", "")
        self.crs = crs
        self.url = f"https://geodata.ucdavis.edu/gadm/gadm{version}/shp/gadm{self._nodot_version}_{iso3}_shp.zip"
        self.path = geoglue.data_path / f"gadm{self._nodot_version}" / iso3
        if not self.path.exists():
            self.path.mkdir(parents=True)

    def fetch(self) -> bool:
        return download_file(self.url, self.path / self.url.split("/")[-1])

    @staticmethod
    def list_admin_cols(i) -> list[str]:
        cols = []
        for k in range(1, i + 1):
            cols.extend(
                [
                    it + "_" + str(k)
                    for it in ["GID", "NAME"]
                ]
            )
        return cols

    @cache
    def __getitem__(self, admin_level: int):
        if admin_level > 3:
            raise IndexError("Only admin level upto 3 supported in GADM")
        return gpd.read_file(
            self.path / f"gadm{self._nodot_version}_{self.iso3}_{admin_level}.shp"
        ).to_crs(self.crs)
