from pathlib import Path
import pytest

import pytz
import numpy as np

from geoglue.country import Country

DATA_PATH = Path("data")


@pytest.fixture(scope="module")
def country():
    return Country("VNM", backend="geoboundaries", data_path=DATA_PATH)


# GADM data cannot be redistributed, and we want to avoid network calls in unit tests
# Will check everything other than shapefile download
@pytest.fixture(scope="module")
def country_gadm():
    return Country("VNM", fetch_data=False)


def test_timezone_assignment():
    cc = Country("VNM", timezone=pytz.timezone("Asia/Ho_Chi_Minh"))
    assert cc.timezone == pytz.timezone("Asia/Ho_Chi_Minh")


def test_invalid_backend():
    with pytest.raises(ValueError, match="Unsupported geographic data backend"):
        Country("VNM", backend="osm")  # type: ignore


def test_timezone_warnings():
    with pytest.warns(match="Multiple timezones for ISO3 country code"):
        cc = Country("USA", fetch_data=False)
        assert cc.timezone == pytz.timezone("America/New_York")


def test_timezone_offset(country, country_gadm):
    assert country.timezone_offset == country_gadm.timezone_offset == "+07:00"


def test_iso2(country, country_gadm):
    assert country.iso2 == country_gadm.iso2 == "VN"


def test_url_gadm(country_gadm):
    assert (
        country_gadm.url
        == "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_VNM_shp.zip"
    )


def test_admin_cols(country, country_gadm):
    assert country_gadm.admin_cols(2) == ["GID_1", "GID_2", "NAME_1", "NAME_2"]
    assert country.admin_cols(2) == ["shapeID", "shapeName"]


def test_invalid_admin_raises_error(country, country_gadm):
    with pytest.raises(ValueError, match="Unsupported adm"):
        country_gadm.admin(4)
    with pytest.raises(ValueError, match="Unsupported adm"):
        country.admin(3)


@pytest.mark.parametrize("year,population", [(2000, 79910432), (2020, 97338600)])
def test_population(country, year, population):
    assert int(country.population_raster(year).sum()) == population


def test_population_invalid_year(country):
    err = "Current population source 'worldpop' only has data from 2000-2020"
    with pytest.raises(ValueError, match=err):
        country.population_raster(1999)
    with pytest.raises(ValueError, match=err):
        country.population_raster(2040)


# Requires shapefiles -- only test with geoboundaries
@pytest.mark.parametrize("adm", [1, 2])
def test_read_shapefiles(country, adm):
    adm_cols = set(country.admin_cols(adm)) | {"geometry"}
    assert adm_cols <= set(country.admin(adm).columns)


def test_bounds(country):
    expected_bounds = np.array(
        [23.39357409300004, 102.14367612700005, 8.403122515000064, 109.46302527100005]
    )
    assert np.allclose(np.array(country.bounds), expected_bounds)
