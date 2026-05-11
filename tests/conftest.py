from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geoglue.types import CdoGriddes
from geoglue.util import read_geotiff

DATA_PATH = Path("data")


@pytest.fixture(scope="session")
def adm2_polygons() -> gpd.GeoDataFrame:
    return gpd.read_file(
        DATA_PATH / "vnm_adm_gov_20201027" / "vnm_admbnda_adm2_gov_20201027.shp"
    )


@pytest.fixture(scope="session")
def population_1km(adm2_polygons) -> CdoGriddes:
    return CdoGriddes.from_dataset(
        read_geotiff(
            DATA_PATH / "VNM" / "worldpop" / "vnm_ppp_2020_1km_Aggregated_UNadj.tif"
        )
        .rio.clip(
            adm2_polygons.geometry,
            adm2_polygons.crs,
            drop=True,
            invert=False,
        )
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
