import datetime
from pathlib import Path

import pytest
import numpy as np
from unittest.mock import patch

from geoglue.memoryraster import MemoryRaster
from geoglue.region import gadm, geoboundaries, get_timezone, Region, get_region
from geoglue.types import Bbox

DATA_PATH = Path("data")
REGION_FILE = Path("tests/data/regions.toml")
EXAMPLE_REGION = Region(
    "gb:VNM",
    {2: Path("data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp")},
    {2: "shapeID"},
    "+07:00",
    "https://www.geoboundaries.org/api/current/gbOpen/VNM/",
    Bbox(
        maxy=24,
        minx=102,
        miny=7,
        maxx=118,
    ),
    None,
)


EXAMPLE_REGION_WITH_ADMIN = Region(
    "gb:VNM",
    {2: Path("data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp")},
    {2: "shapeID"},
    "+07:00",
    "https://www.geoboundaries.org/api/current/gbOpen/VNM/",
    Bbox(
        maxy=24,
        minx=102,
        miny=7,
        maxx=118,
    ),
    2,
)
EXAMPLE_REGION_STRING = (
    "gb:VNM 102,7,118,24 +07:00 https://www.geoboundaries.org/api/current/gbOpen/VNM/"
)


def test_region_to_string():
    assert str(EXAMPLE_REGION) == EXAMPLE_REGION_STRING


@pytest.fixture(scope="module")
def region_geoboundaries():
    return geoboundaries("VNM", data_path=DATA_PATH)


@pytest.fixture(scope="module")
@patch("geoglue.util.download_file", return_value=True, autospec=True)
def region_gadm(_):
    return gadm("VNM")


def test_region_geoboundaries(region_geoboundaries):
    assert region_geoboundaries == Region(
        "gb:VNM",
        {
            1: Path("data/VNM/geoboundaries/geoBoundaries-VNM-ADM1.shp"),
            2: Path("data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp"),
        },
        "shapeID",
        "+07:00",
        "https://www.geoboundaries.org/api/current/gbOpen/VNM/",
        Bbox(
            maxy=23.39097449667571,
            minx=102.14588283092797,
            miny=8.43995762984082,
            maxx=114.99696468067555,
        ),
    )


def test_region_gadm(region_gadm):
    assert region_gadm == Region(
        "gadm:VNM",
        {
            i: Path.home() / f".local/share/geoglue/VNM/gadm41/gadm41_VNM_{i}.shp"
            for i in range(4)
        },
        {i: f"GID_{i}" for i in range(4)},
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


@pytest.mark.parametrize("year,population", [(2000, 79910432), (2020, 97338600)])
def test_worldpop_1km(year, population):
    rast = MemoryRaster.read(
        f"data/VNM/worldpop/vnm_ppp_{year}_1km_Aggregated_UNadj.tif"
    )
    assert int(rast.sum()) == population


def test_read_shapefiles(region_geoboundaries):
    assert {"shapeID", "shapeName"} <= set(region_geoboundaries.read_admin(1).columns)


def test_bounds(region_geoboundaries):
    expected_bounds = np.array([102.14588283, 8.43995763, 114.99696468, 23.3909745])
    actual_bounds = np.array(region_geoboundaries.bbox)
    assert np.allclose(actual_bounds, expected_bounds)


@pytest.mark.parametrize(
    "region_name,region",
    [("gb:VNM", EXAMPLE_REGION), ("gb:VNM-2", EXAMPLE_REGION_WITH_ADMIN)],
)
def test_valid_get_region(region_name, region):
    assert get_region(region_name, REGION_FILE) == region


@pytest.mark.parametrize("region", ["invalid_tz", "invalid_bounds"])
def test_invalid_get_region(region):
    with pytest.raises(ValueError):
        get_region(region, REGION_FILE)
