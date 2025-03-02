import datetime
from pathlib import Path

import pytest
import xarray as xr
from geoglue.zonal_stats import DatasetZonalStatistics
from geoglue.country import Country
from geoglue.resample import resample
from geoglue.types import CdoResampling

DATA_PATH = Path("data")
PRECIP_DATA = DATA_PATH / "VNM" / "era5" / "VNM-2020-era5.daily_sum.nc"
ADMIN2_N = 706


@pytest.fixture(scope="module")
def vnm_geoboundaries_admin2():
    cc = Country("VNM", backend="geoboundaries", data_path=DATA_PATH)
    df = cc.admin(2)

    # We use era5_extents which use admin0 boundaries. In geoBoundaries data,
    # the admin0 total_bounds are smaller than the admin2 total bounds which
    # includes the archipelago of Huong Sa and Truong Sa. As the downloaded
    # ERA5 data is according to the admin0 extents, this will lead to NA values
    # in the weather raster for these administrative regions, so we remove these
    return df[~df.shapeName.isin(["Hoang Sa", "Truong Sa"])]


@pytest.fixture(scope="module")
def vnm_pop():
    return Country("VNM").population_raster(2020)


@pytest.fixture(scope="module")
def dataset(vnm_geoboundaries_admin2, vnm_pop):
    ds = xr.open_dataset(PRECIP_DATA)
    return DatasetZonalStatistics(ds, vnm_geoboundaries_admin2, vnm_pop)


@pytest.fixture(scope="module")
def dataset_unweighted(vnm_geoboundaries_admin2):
    ds = xr.open_dataset(PRECIP_DATA)
    return DatasetZonalStatistics(ds, vnm_geoboundaries_admin2)


@pytest.fixture(scope="module")
def dataset_resampled(vnm_geoboundaries_admin2, vnm_pop):
    outfile = PRECIP_DATA.parent / (PRECIP_DATA.stem + "_remapdis.nc")
    if not outfile.exists():
        resample(CdoResampling.remapdis, PRECIP_DATA, vnm_pop, outfile)
    ds = xr.open_dataset(outfile)
    return DatasetZonalStatistics(ds, vnm_geoboundaries_admin2, vnm_pop)


def test_dataset_properties(dataset):
    assert dataset.time_col == "valid_time"
    # longitude should not be from 180 - 360
    assert float(dataset.dataset.longitude.max()) < 180
    assert dataset.variables == ["tp"]


def test_zonal_stats_raises_error(dataset):
    d = datetime.date(2020, 1, 1)
    with pytest.raises(ValueError, match="Variable shape"):
        dataset.zonal_stats("tp", "sum", min_date=d, max_date=d)


@pytest.mark.parametrize(
    "ds,ops,int_value_max",
    [
        ("dataset_resampled", "sum", 2274),
        ("dataset_resampled", "area_weighted_sum", 45),
        ("dataset_unweighted", "sum", 0),
    ],
)
def test_zonal_stats(ds, ops, int_value_max, request):
    ds = request.getfixturevalue(ds)
    d = datetime.date(2020, 1, 1)
    df = ds.zonal_stats("tp", ops, min_date=d, max_date=d)
    assert len(df) == ADMIN2_N
    assert (df.value >= 0).all()  # non-negative values

    assert int(df.value.max()) == int_value_max
    if ops == "area_weighted_sum":
        assert "weighted_sum" in df.columns and "count" in df.columns
