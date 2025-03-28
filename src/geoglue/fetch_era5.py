"""
ERA5 fetch module

Requires a cdsapi account, see https://cds.climate.copernicus.eu/how-to-api
for instructions on obtaining an API key. The API key needs to be written
to $HOME/.cdsapirc with the following contents:

    url: https://cds.climate.copernicus.eu/api
    key: <PERSONAL-ACCESS-TOKEN>
"""

from pathlib import Path

import cdsapi
from .util import zero_padded_intrange
from .country import Country

SUPPORTED_STATISTICS = ["daily_mean", "daily_min", "daily_max", "daily_sum"]
MONTHS = zero_padded_intrange(1, 12)
DAYS = zero_padded_intrange(1, 31)
TIMES = [s + ":00" for s in zero_padded_intrange(0, 23)]


def fetch_era5_daily(
    iso3: str,
    year: int,
    variables: list[str],
    statistic: str,
    frequency: str = "6_hourly",
) -> Path:
    cc = Country(iso3)
    if statistic not in SUPPORTED_STATISTICS:
        raise ValueError(
            f"fetch_era5_daily: {statistic=} not in supported list {SUPPORTED_STATISTICS}"
        )
    dataset = "derived-era5-single-levels-daily-statistics"
    offset = cc.timezone_offset
    request = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": year,
        "month": MONTHS,
        "day": DAYS,
        "daily_statistic": statistic,
        "time_zone": f"utc{offset}",
        "frequency": frequency,
        "area": list(cc.integer_bounds)
    }

    client = cdsapi.Client()
    output = f"{iso3.upper()}-{year}-era5.{statistic}.nc"
    client.retrieve(dataset, request, output)
    return Path(output)


def fetch_era5_hourly(iso3: str, year: int, variables: list[str]) -> Path:
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [year],
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": Country(iso3).era5_extents,
    }

    client = cdsapi.Client()
    output = "{iso3.upper()}-{year}-era5.hourly.zip"
    client.retrieve(dataset, request, output).download()
    return Path(output)
