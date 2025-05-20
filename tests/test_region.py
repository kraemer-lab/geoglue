import datetime
from pathlib import Path

import pytest
import numpy as np
from unittest.mock import patch

from geoglue.region import (
    gadm,
    geoboundaries,
    get_timezone,
    get_worldpop_1km,
    read_region,
)
from geoglue.types import Bounds

DATA_PATH = Path("data")


@pytest.fixture(scope="module")
def region_geoboundaries():
    return geoboundaries("VNM", 2, data_path=DATA_PATH)


@pytest.fixture(scope="module")
@patch("geoglue.util.download_file", return_value=True, autospec=True)
def region_gadm(_):
    return gadm("VNM", 2)


def test_region_geoboundaries(region_geoboundaries):
    assert region_geoboundaries == {
        "path": Path("data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp"),
        "name": "VNM-2",
        "tz": "+07:00",
        "pk": "shapeID",
        "url": "https://www.geoboundaries.org/api/current/gbOpen/VNM/",
        "bounds": Bounds(
            north=23.392205570000044,
            west=102.14402486200004,
            south=7.180931477000058,
            east=117.83545743800005,
        ),
    }


def test_region_gadm(region_gadm):
    assert region_gadm == {
        "path": Path.home() / ".local/share/geoglue/VNM/gadm41/gadm41_VNM_2.shp",
        "name": "VNM-2",
        "tz": "+07:00",
        "pk": "GID_2",
        "url": "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_VNM_shp.zip",
        "bounds": Bounds(
            north=23.39269256700004,
            west=102.14458465500007,
            south=8.381355000000099,
            east=109.46917000000008,
        ),
    }


def test_timezone_warnings():
    assert get_timezone("USA", datetime.datetime(2024, 1, 1)) is None


def test_invalid_admin_raises_error():
    with pytest.raises(ValueError, match="Unsupported administrative level"):
        gadm("VNM", 4)
    with pytest.raises(ValueError, match="Unsupported administrative level"):
        geoboundaries("VNM", 3)


@pytest.mark.parametrize("year,population", [(2000, 79910432), (2020, 97338600)])
def test_worldpop_1km(year, population):
    assert int(get_worldpop_1km("VNM", year, data_path=DATA_PATH).sum()) == population


def test_population_invalid_year():
    err = "Worldpop population data is only available from 2000-2020"
    with pytest.raises(ValueError, match=err):
        get_worldpop_1km("VNM", 1999)
    with pytest.raises(ValueError, match=err):
        get_worldpop_1km("VNM", 2040)


def test_read_shapefiles(region_geoboundaries):
    shp = read_region(region_geoboundaries)
    assert {"shapeID", "shapeName"} <= set(shp.columns)


def test_bounds(region_geoboundaries):
    expected_bounds = np.array([23.39220557, 102.14402486, 7.18093148, 117.83545744])
    actual_bounds = np.array(region_geoboundaries["bounds"])
    print(actual_bounds)
    assert np.allclose(actual_bounds, expected_bounds)
