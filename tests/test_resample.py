import xarray as xr

from geoglue.resample import resample, resampled_dataset
from geoglue.memoryraster import MemoryRaster

import pytest

INFILE = "data/VNM/era5/VNM-2020-era5.daily_sum.nc"


@pytest.mark.parametrize("resampling", ["remapbil", "remapdis"])
def test_resample(resampling, population_1km):
    outfile = resample(resampling, INFILE, target=population_1km, skip_exists=False)
    outrast = MemoryRaster.from_xarray(xr.open_dataset(outfile).tp.isel(valid_time=0))
    assert outrast.griddes.approx_equal(population_1km.griddes)
    outfile.unlink()


@pytest.mark.parametrize("resampling", ["remapbil", "remapdis"])
def test_resampled_dataset(resampling, population_1km):
    with resampled_dataset(resampling, INFILE, target=population_1km) as ds:
        assert (
            ds.attrs["provenance"]
            == "resample.infile=sha256:e3c164a6cc43ab63d7707fd42115f8f26d55e4944a9ab61ec9039070c4ea392d data/VNM/era5/VNM-2020-era5.daily_sum.nc\n"
        )
        outrast = MemoryRaster.from_xarray(ds.tp.isel(valid_time=0))
        assert outrast.griddes.approx_equal(population_1km.griddes)
