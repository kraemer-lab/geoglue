from pathlib import Path
import pytest


from geoglue.country import Country

DATA_PATH = Path("data")


@pytest.fixture(scope="module")
def country():
    return Country("VNM", backend="geoboundaries", data_path=DATA_PATH)


# GADM data cannot be redistributed, and we want to avoid network calls in unit tests
# Will check everything other than shapefile download
@pytest.fixture(scope="module")
def country_gadm():
    return Country("VNM")


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
    assert country.admin_cols(2) == ["shapeID", "shapeName", "shapeISO"]


def test_invalid_admin_raises_error(country, country_gadm):
    with pytest.raises(ValueError, match="Unsupported adm"):
        country_gadm.admin(4)
    with pytest.raises(ValueError, match="Unsupported adm"):
        country.admin(3)


@pytest.mark.parametrize("year,population", [(2000, 79910432), (2020, 97338600)])
def test_population(country, year, population):
    assert int(country.population_raster(year).sum()) == population


# Requires shapefiles -- only test with geoboundaries
@pytest.mark.parametrize("adm", [1, 2])
def test_read_shapefiles(country, adm):
    adm_cols = set(country.admin_cols(adm)) | {"geometry"}
    assert adm_cols <= set(country.admin(adm).columns)


def test_era5_extents(country):
    assert country.era5_extents == [24, 102, 8, 110]
