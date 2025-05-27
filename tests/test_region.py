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
    Region,
)
from geoglue.types import Bbox

DATA_PATH = Path("data")

EXAMPLE_REGION = Region(
    "gb:VNM-2",
    Path("data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp"),
    "shapeID",
    "+07:00",
    "https://www.geoboundaries.org/api/current/gbOpen/VNM/",
    Bbox(
        maxy=24,
        minx=102,
        miny=7,
        maxx=118,
    ),
)

EXAMPLE_REGION_STRING = "gb:VNM-2 102,7,118,24 shapeID +07:00 data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp https://www.geoboundaries.org/api/current/gbOpen/VNM/"


def test_region_to_string():
    assert str(EXAMPLE_REGION) == EXAMPLE_REGION_STRING


def test_region_from_string():
    assert Region.from_string(EXAMPLE_REGION_STRING) == EXAMPLE_REGION


@pytest.fixture(scope="module")
def region_geoboundaries():
    return geoboundaries("VNM", 2, data_path=DATA_PATH)


@pytest.fixture(scope="module")
@patch("geoglue.util.download_file", return_value=True, autospec=True)
def region_gadm(_):
    return gadm("VNM", 2)


def test_region_geoboundaries(region_geoboundaries):
    assert region_geoboundaries == Region(
        "gb:VNM-2",
        Path("data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp"),
        "shapeID",
        "+07:00",
        "https://www.geoboundaries.org/api/current/gbOpen/VNM/",
        Bbox(
            maxy=23.392205570000044,
            minx=102.14402486200004,
            miny=7.180931477000058,
            maxx=117.83545743800005,
        ),
    )


def test_region_gadm(region_gadm):
    assert region_gadm == Region(
        "gadm:VNM-2",
        Path.home() / ".local/share/geoglue/VNM/gadm41/gadm41_VNM_2.shp",
        "GID_2",
        "+07:00",
        "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_VNM_shp.zip",
        Bbox(
            maxy=23.39269256700004,
            minx=102.14458465500007,
            miny=8.381355000000099,
            maxx=109.46917000000008,
        ),
    )


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
    assert {"shapeID", "shapeName"} <= set(region_geoboundaries.read().columns)


def test_bounds(region_geoboundaries):
    expected_bounds = np.array([102.14402486, 7.18093148, 117.83545744, 23.39220557])
    actual_bounds = np.array(region_geoboundaries.bbox)
    print(actual_bounds)
    assert np.allclose(actual_bounds, expected_bounds)
