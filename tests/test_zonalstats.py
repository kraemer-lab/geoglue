from pathlib import Path

import pytest
import xarray as xr
from geoglue.types import CdoGriddes
from geoglue.zonalstats import zonalstats
from geoglue.region import geoboundaries
from geoglue.resample import resample
from geoglue.util import read_geotiff, sort_lonlat

DATA_PATH = Path("data")
PRECIP_DATA = DATA_PATH / "VNM" / "era5" / "VNM-2020-era5.daily_sum.nc"
ADMIN2_N = 706


@pytest.fixture(scope="module")
def vnm_geoboundaries_admin2():
    df = geoboundaries("VNM", data_path=DATA_PATH).read_admin(2)

    # We use era5_extents which use admin0 boundaries. In geoBoundaries data,
    # the admin0 total_bounds are smaller than the admin2 total bounds which
    # includes the archipelago of Huong Sa and Truong Sa. As the downloaded
    # ERA5 data is according to the admin0 extents, this will lead to NA values
    # in the weather raster for these administrative regions, so we remove these
    return df[~df.shapeName.isin(["Hoang Sa", "Truong Sa"])].set_index("shapeID")


@pytest.fixture(scope="module")
def vnm_pop():
    return read_geotiff(Path("data/VNM/worldpop/vnm_ppp_2020_1km_Aggregated_UNadj.tif"))


@pytest.fixture(scope="module")
def dataset():
    ds = sort_lonlat(xr.open_dataset(PRECIP_DATA))
    return ds


@pytest.fixture(scope="module")
def dataset_resampled(vnm_pop):
    griddes = CdoGriddes.from_dataset(vnm_pop)
    outfile = PRECIP_DATA.parent / (PRECIP_DATA.stem + "_remapdis.nc")
    print("Resampled dataset =", resample("remapdis", PRECIP_DATA, griddes, outfile))
    ds = xr.open_dataset(outfile)
    yield ds
    if outfile.exists():
        outfile.unlink()


def test_zonal_stats_raises_error(dataset, vnm_geoboundaries_admin2, vnm_pop):
    da = dataset.tp.sel(valid_time=slice("2020-01-01", "2020-01-01"))
    with pytest.raises(RuntimeError, match="Incompatible extent"):
        zonalstats(da, vnm_geoboundaries_admin2, "sum", weights=vnm_pop)


@pytest.mark.parametrize(
    "op,int_value_max",
    [
        ("weighted_sum(default_value=0,default_weight=0)", 2269),
        ("area_weighted_sum(default_value=0,default_weight=0)", 45),
        ("area_weighted_sum(default_weight=0)", 45),
    ],
)
def test_zonalstats(
    dataset_resampled, vnm_geoboundaries_admin2, vnm_pop, op, int_value_max
):
    da = dataset_resampled.tp.sel(valid_time=slice("2020-01-01", "2020-01-01"))
    za = zonalstats(da, vnm_geoboundaries_admin2, op, weights=vnm_pop)
    assert len(za.region) == ADMIN2_N
    assert (za >= 0).all()  # non-negative values
    assert int(za.max().item()) == int_value_max
