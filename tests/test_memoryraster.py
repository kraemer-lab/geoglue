"Tests for geoglue.memoryraster module"

from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.enums import Resampling
import pytest

from geoglue.types import CdoGriddes
from geoglue.memoryraster import get_numpy_dtype, MemoryRaster

DATA_PATH = Path("data")


@pytest.fixture(scope="module")
def aedes(adm2_polygons):
    return MemoryRaster.read(DATA_PATH / "aegypti.tif", crs=adm2_polygons.crs.srs).mask(
        adm2_polygons
    )


def test_aedes_checksum(aedes):
    assert (
        aedes.checksum()
        == "MemoryRaster.origin_path=sha256:64f38840ad4d349fb36f79ae699bfa9a339769c188797a6ff8ec0a7ac2a5a24d data/aegypti.tif 378x390"
    )


@pytest.fixture(scope="module")
def population_lowres(population_1km, adm2_polygons):
    aedes = MemoryRaster.read(
        DATA_PATH / "aegypti.tif", crs=adm2_polygons.crs.srs
    ).mask(adm2_polygons)
    return population_1km.resample(aedes, Resampling.sum)


def test_get_numpy_dtype():
    assert get_numpy_dtype("float32") == np.float32
    assert get_numpy_dtype("any") == np.float64  # default return type


def test_population_1km(population_1km):
    assert population_1km.shape == (1781, 879)
    assert int(population_1km.sum()) == 96765864
    assert int(population_1km.min()) == 0
    assert int(population_1km.max()) == 69419


def test_resample_and_lowres_sum(population_lowres, aedes):
    # check that low resolution population is sampled correctly
    assert aedes.shape == population_lowres.shape == (390, 378)

    # Unless the new grid is exactly an integer scaling of the old grid, it is
    # be expected that the population counts will not be exactly the same, as
    # we use bilinear interpolation to assign values to grid points. Here
    # we are off by 16 from the 1km population sum
    assert int(population_lowres.sum()) == 96765880


def test_zonal_stats(aedes, adm2_polygons, population_lowres):
    df = aedes.zonal_stats(
        adm2_polygons,
        "weighted_mean",
        population_lowres,
        include_cols=["ADM2_EN", "ADM1_EN"],
    )

    # TODO: Unclear what the expected result would be, so we only check that
    # exact_extract ran properly
    assert list(df.columns) == ["ADM2_EN", "ADM1_EN", "weighted_mean"]
    assert pd.notna(df.weighted_mean).any()

    # TODO: Check population of Ho Chi Minh city, estimate returned is higher
    # than expected. Accounting for urban and rural areas, it is expected to be
    # about 9.4m, whereas we get 10.4m
    pop = population_lowres.zonal_stats(
        adm2_polygons, "sum", include_cols=["ADM2_EN", "ADM1_EN"], include_geom=True
    )
    pop_hcmc = int(pop[pop.ADM1_EN == "Ho Chi Minh city"]["sum"].sum())
    assert pop_hcmc == 10365660


def test_griddes(population_1km):
    assert population_1km.griddes == CdoGriddes(
        gridtype="lonlat",
        gridsize=1565499,
        xsize=879,
        ysize=1781,
        xname="longitude",
        yname="latitude",
        ylongname="latitude",
        yunits="degrees_north",
        xfirst=102.14874960275003,
        xinc=0.0083333333,
        yfirst=8.557916831327146,
        yinc=0.0083333333,
        xlongname="longitude",
        xunits="degrees_east",
    )
