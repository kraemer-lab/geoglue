from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

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
            DATA_PATH / "VNM" / "worldpop" / "vnm_ppp_2020_1km_Aggregated_UNadj.tif",
            crs=adm2_polygons.crs.srs,
        )
        .mask(adm2_polygons)
        .astype(np.float32)
    )


@pytest.fixture(scope="session")
def temp_dataset():
    lat = [10, 20]
    lon = [30, 40]
    time = pd.date_range("2023-01-01", periods=3)

    # Create some dummy data
    data = np.random.rand(len(time), len(lat), len(lon))

    # Create the dataset
    return xr.Dataset(
        {"temperature": (["time", "latitude", "longitude"], data)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


@pytest.fixture(scope="session")
def temp_dataset_multiple_time_axes(temp_dataset):
    valid_time = temp_dataset.time + pd.to_timedelta("6h")
    return temp_dataset.assign_coords(valid_time=valid_time)
