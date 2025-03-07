import xarray as xr

from geoglue.resample import resample
from geoglue.memoryraster import MemoryRaster

import pytest


@pytest.mark.parametrize("resampling", ["remapbil", "remapdis"])
def test_resample(resampling, population_1km):
    infile = "data/VNM/era5/VNM-2020-era5.daily_sum.nc"
    outfile = resample(resampling, infile, target=population_1km, skip_exists=False)
    outrast = MemoryRaster.from_xarray(xr.open_dataset(outfile).tp.isel(valid_time=0))
    assert outrast.griddes.approx_equal(population_1km.griddes)
    outfile.unlink()
