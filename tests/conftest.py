from pathlib import Path

import pytest
import numpy as np
import geopandas as gpd

from geoglue.memoryraster import MemoryRaster

DATA_PATH = Path("data")


@pytest.fixture(scope="session")
def adm2_polygons():
    return gpd.read_file(
        DATA_PATH / "vnm_adm_gov_20201027" / "vnm_admbnda_adm2_gov_20201027.shp"
    )


@pytest.fixture(scope="session")
def population_1km(adm2_polygons):
    return (
        MemoryRaster.read(
            DATA_PATH / "vnm_ppp_2020_1km_Aggregated_UNadj.tif", crs=adm2_polygons.crs.srs
        )
        .mask(adm2_polygons)
        .astype(np.float32)
    )
