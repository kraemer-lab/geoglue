import xarray as xr

from geoglue.resample import resample, resampled_dataset
from geoglue.memoryraster import MemoryRaster
from geoglue.util import read_geotiff
from geoglue.types import CdoGriddes

import pytest

INFILE = "data/VNM/era5/VNM-2020-era5.daily_sum.nc"


def na_frac(da: xr.DataArray) -> float:
    return da.isnull().sum().item() / da.size


@pytest.mark.parametrize("resampling", ["remapbil", "remapdis"])
def test_resample(resampling, population_1km):
    outfile = resample(resampling, INFILE, target=population_1km, skip_exists=False)
    outrast = MemoryRaster.from_xarray(xr.open_dataset(outfile).tp.isel(valid_time=0))
    assert outrast.griddes.approx_equal(population_1km.griddes)
    outfile.unlink()


@pytest.mark.parametrize("resampling", ["remapbil", "remapdis"])
def test_resampled_dataset(resampling, population_1km):
    with resampled_dataset(resampling, INFILE, target=population_1km) as ds:
        outrast = MemoryRaster.from_xarray(ds.tp.isel(valid_time=0))
        assert outrast.griddes.approx_equal(population_1km.griddes)


def test_resample_sparse_vs_resample():
    weights = read_geotiff("data/SGP/sgp_pop_2015_CN_1km_R2025A_UA_v1.tif")
    # source = xr.open_dataarray("data/SGP/SGP-ndvi-2015.nc")
    griddes = CdoGriddes.from_dataset(weights)
    remapbil = xr.open_dataarray(
        resample("remapbil", "data/SGP/SGP-ndvi-2015_na.nc", griddes, skip_exists=False)
    )
    sremapbil = xr.open_dataarray(
        resample(
            "sremapbil", "data/SGP/SGP-ndvi-2015_na.nc", griddes, skip_exists=False
        )
    )

    # Resampling method should not affect shape
    assert remapbil.shape == sremapbil.shape == (37, 59)
    remapbil_nna = ~remapbil.isnull()
    sremapbil_nna = ~sremapbil.isnull()

    # sremapbil should have less NAs as it handles NA/non-NA edges
    assert na_frac(sremapbil) < na_frac(remapbil)

    # Any coordinate which is non-NA in remapbil should also
    # be non-NA in sremapbil
    assert (remapbil_nna & ~sremapbil_nna).sum().item() == 0
