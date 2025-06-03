import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

from geoglue.cds import (
    ReanalysisSingleLevels,
    era5_extract_hourly_data,
    CdsPath,
    get_timezone_offset_hours,
    timeshift_hours_cdsdataset,
    DatasetPool,
    grib_to_netcdf,
)
from geoglue.region import Region
from geoglue.types import Bbox

BRB = Region(
    "BRB-1",
    "/path/to/BRB.shp",
    "GID_1",
    "-04:00",
    "https://gadm.org",
    Bbox(maxy=14, minx=-59, miny=13, maxx=-58),
)

SGP = Region(
    "SGP-1",
    "/path/to/SGP.shp",
    "GID_1",
    "+08:00",
    "https://gadm.org",
    Bbox(maxy=2, minx=103, miny=1, maxx=105),
)

NPL = Region(
    "NPL-1",
    "/path/to/NPL.shp",
    "GID_1",
    "+05:45",
    "https://gadm.org",
    Bbox(
        minx=80.0884245137, miny=26.3978980576, maxx=88.1748043151, maxy=30.4227169866
    ),
)


# fmt: off
MONTHS =  ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
TIMES = [
    "00:00", "01:00", "02:00",
    "03:00", "04:00", "05:00",
    "06:00", "07:00", "08:00",
    "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00",
    "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00",
    "21:00", "22:00", "23:00"
]
DAYS =  [
    "01", "02", "03", "04", "05", "06", "07",
    "08", "09", "10", "11", "12", "13", "14",
    "15", "16", "17", "18", "19", "20", "21",
    "22", "23", "24", "25", "26", "27", "28",
    "29", "30", "31"
]
# fmt: on

VARIABLES = ["2m_temperature", "total_precipitation"]
EXPECTED_REQUEST_NETCDF = {
    "product_type": ["reanalysis"],
    "variable": VARIABLES,
    "year": ["2020"],
    "month": MONTHS,
    "day": DAYS,
    "time": TIMES,
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [2, 103, 1, 105],
}

EXPECTED_REQUEST_GRIB = {
    "product_type": ["reanalysis"],
    "variable": VARIABLES,
    "year": ["2020"],
    "month": MONTHS,
    "day": DAYS,
    "time": TIMES,
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [2, 103, 1, 105],
}


@pytest.mark.parametrize(
    "offset,hours", [("+05:00", 5), ("-04:00", -4), ("+01:30", None)]
)
def test_get_timezone_offset_hours(offset, hours):
    assert get_timezone_offset_hours(offset) == hours


@pytest.fixture(scope="module")
def data_singapore():
    return ReanalysisSingleLevels(
        SGP,
        VARIABLES,
        path=Path("tests/data"),
        data_format="netcdf",
    )


@pytest.fixture(scope="module")
def data_barbados():
    return ReanalysisSingleLevels(
        BRB,
        VARIABLES,
        path=Path("tests/data"),
        data_format="netcdf",
    )


def test_request(data_singapore):
    assert data_singapore._cdsapi_request(2020) == EXPECTED_REQUEST_NETCDF


def test_era5_extract_hourly_data():
    with tempfile.TemporaryDirectory() as folder:
        assert era5_extract_hourly_data(
            Path("tests/data/SGP-2020-era5.zip"), Path(folder)
        ) == CdsPath(
            instant=Path(folder) / "SGP-2020-era5.instant.nc",
            accum=Path(folder) / "SGP-2020-era5.accum.nc",
        )


@patch("cdsapi.Client", autospec=True)
def test_get_when_file_exists(mock_client, data_singapore):
    test_data = Path("tests/data")
    assert data_singapore.get(2020) == CdsPath(
        instant=test_data / "SGP-2020-era5.instant.nc",
        accum=test_data / "SGP-2020-era5.accum.nc",
    )
    # file already exists, so no need to call cdsapi.Client().retrieve
    mock_client().retrieve.assert_not_called()


@patch("cdsapi.Client", autospec=True)
def test_get_netcdf(mock_client):
    ReanalysisSingleLevels(
        SGP,
        VARIABLES,
        data_format="netcdf",
    ).get(2020)
    mock_client().retrieve.assert_called_once_with(
        "reanalysis-era5-single-levels",
        EXPECTED_REQUEST_NETCDF,
        Path("~/.local/share/geoglue/SGP/era5/SGP-2020-era5.zip").expanduser(),
    )


@patch("cdsapi.Client", autospec=True)
def test_get_grib(mock_client):
    ReanalysisSingleLevels(
        SGP,
        VARIABLES,
        data_format="grib",
    ).get(2020)
    mock_client().retrieve.assert_called_once_with(
        "reanalysis-era5-single-levels",
        EXPECTED_REQUEST_GRIB,
        Path("~/.local/share/geoglue/SGP/era5/SGP-2020-era5.grib").expanduser(),
    )


def test_cds_dataset(data_singapore):
    ds = data_singapore.get(year=2020).as_dataset()
    assert ds.is_hourly is True
    assert ds.get_time_dim() == "valid_time"


@pytest.mark.parametrize("kind", ["instant", "accum"])
def test_dataset_pool_positive_time_shift(data_singapore, kind):
    pool = data_singapore.get_dataset_pool()
    ds19 = pool.path(2019).as_dataset()

    # drop valid_time as time coordinates will not match
    shift = 8 if kind == "instant" else 7
    last_2019_hours = (
        getattr(ds19, kind).isel(valid_time=slice(-shift, None)).drop_vars("valid_time")
    )
    first_2020_hours = (
        getattr(pool[2020], kind)
        .isel(valid_time=slice(None, shift))
        .drop_vars("valid_time")
    )
    assert last_2019_hours.equals(first_2020_hours)
    assert getattr(pool[2020], kind).valid_time.min().values == np.datetime64(
        "2020-01-01"
    )


@pytest.mark.parametrize("kind", ["instant", "accum"])
def test_dataset_pool_negative_time_shift(data_barbados, kind):
    pool = data_barbados.get_dataset_pool()
    ds20 = pool.path(2020).as_dataset()

    shift = 4 if kind == "instant" else 5
    # drop valid_time as time coordinates will not match
    first_2020_hours = (
        getattr(ds20, kind).isel(valid_time=slice(None, shift)).drop_vars("valid_time")
    )
    print(first_2020_hours)
    last_2019_hours = (
        getattr(pool[2019], kind)
        .isel(valid_time=slice(-shift, None))
        .drop_vars("valid_time")
    )
    print(last_2019_hours)
    assert last_2019_hours.equals(first_2020_hours)
    assert getattr(pool[2019], kind).valid_time.max().values == np.datetime64(
        "2019-12-31T23"
    )


def test_timeshift_errors(data_barbados):
    pool = data_barbados.get_dataset_pool()
    ds19 = pool.path(2019).as_dataset()
    ds20 = pool.path(2020).as_dataset()

    with pytest.raises(ValueError, match="Timeshift valid for shift=-12..12"):
        timeshift_hours_cdsdataset(ds19, ds20, shift=-13)
    with pytest.raises(ValueError, match="Timeshift valid for shift=-12..12"):
        timeshift_hours_cdsdataset(ds19, ds20, shift=15)


def test_era5_extract_hourly_data_raises_error():
    with pytest.raises(ValueError, match="Not a valid zip"):
        era5_extract_hourly_data(Path("hello.gz"), Path.cwd())


def test_fractional_offset_raises_error():
    cc = ReanalysisSingleLevels(NPL, VARIABLES)
    with pytest.raises(
        ValueError, match="Can't perform timeshift for fractional timezone offset"
    ):
        cc.get_dataset_pool()


def test_missing_year_datasetpool():
    pool_sgp = DatasetPool(Path("tests/data").glob("SGP*era5.*.nc"), shift_hours=8)
    pool_brb = DatasetPool(Path("tests/data").glob("BRB*era5.*.nc"), shift_hours=-4)

    with pytest.raises(
        FileNotFoundError, match="Positive shift_hours=8 require preceding year"
    ):
        pool_sgp[2018]
    with pytest.raises(
        FileNotFoundError, match="Negative shift_hours=-4 require succeeding year"
    ):
        pool_brb[2020]


def test_multiple_iso3_datasetpool_raises_error():
    with pytest.raises(
        ValueError,
        match="Multiple iso3=.* or stubs=.* not allowed in DatasetPool, specify a stricter path glob",
    ):
        DatasetPool(Path("tests/data").glob("*.nc"), shift_hours=8)


def test_zero_shift_datasetpool():
    pool = DatasetPool(Path("tests/data").glob("SGP*era5.*.nc"), shift_hours=0)
    assert pool[2019] == pool.path(2019).as_dataset()


def test_daily_statistics():
    pool_sgp = DatasetPool(Path("tests/data").glob("SGP*era5.*.nc"), shift_hours=8)
    data = pool_sgp[2020]
    daily = data.daily()

    assert daily.instant.valid_time.size == data.instant.valid_time.size / 24
    assert daily.accum.valid_time.size == data.accum.valid_time.size / 24

    daily_max = data.daily_max()
    daily_min = data.daily_min()
    assert (daily_min < daily.instant).all()
    assert (daily.instant < daily_max).all()


def test_grib_netcdf_match():
    netcdf = CdsPath(
        instant=Path("tests/data/SGP-2019-era5.instant.nc"),
        accum=Path("tests/data/SGP-2019-era5.accum.nc"),
    ).as_dataset()
    grib = grib_to_netcdf(
        Path("tests/data/SGP-2019-era5_grib.grib"), Path("tests/data")
    ).as_dataset()

    for t in ["instant", "accum"]:
        assert getattr(grib, t).equals(getattr(netcdf, t))
        # grib_t = getattr(grib, t).drop_vars(["number", "surface"])
        # netcdf_t = getattr(netcdf, t).drop_vars(["number", "expver"])
        # assert grib_t.equals(netcdf_t)


@pytest.mark.parametrize("window", [0, 6])
def test_weekly_reduce(data_singapore, window):
    pool = data_singapore.get_dataset_pool()
    ds = pool.weekly_reduce(2020, "instant", window=window)
    diff = ds.valid_time[1] - ds.valid_time[0]
    assert int(diff / 1e9) == 7 * 24 * 3600  # check that dataset is weekly
    # assert first and last dates should align to W-MON, closed on the left
    match window:
        case 0:
            assert ds.valid_time.min() == np.datetime64("2020-01-06")
        case 6:
            assert ds.valid_time.min() == np.datetime64("2019-11-25")
    assert ds.valid_time.max() == np.datetime64("2020-12-28")
