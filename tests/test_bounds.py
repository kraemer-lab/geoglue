"Test bounds comparison"

import xarray as xr
import numpy.testing as npt
from geoglue.types import Bbox

b1 = Bbox(maxy=10, minx=-5, miny=-1, maxx=20)
b2 = Bbox(maxy=8, minx=-3, miny=0, maxx=15)
b = Bbox(maxy=23.5, minx=102.20, miny=8.1, maxx=109.89)

# not intersecting
b3 = Bbox(maxy=9, minx=-7, miny=1, maxx=15)


def test_area():
    new_mexico_bounds = Bbox(-109.1, 31.3, -103.0, 37.0)
    npt.assert_approx_equal(new_mexico_bounds.geodetic_area_km2, 355559.9140749907)


def test_bounds_enclosed():
    assert b2 <= b1
    assert b1 >= b2


def test_bounds_not_enclosed():
    assert not (b1 <= b3)


def test_intersection():
    assert b1 & b3 == Bbox(-5, 1, 15, 9)


def test_intersection_empty():
    assert b1 & Bbox(-10, -1, -5, 5) is None


def test_integer_bounds():
    assert b.int() == Bbox(102, 8, 110, 24)


def test_from_string():
    assert Bbox.from_string("102.20,8.1,109.89,23.5") == b


def test_to_list():
    assert b.to_list("cdsapi") == [23.5, 102.20, 8.1, 109.89]
    assert b.int().to_list("cdsapi") == [24, 102, 8, 110]


def test_xarray_bounds():
    ds = xr.open_dataset("tests/data/BRB-2020-era5.instant.nc")
    assert Bbox.from_xarray(ds) == Bbox(minx=-59.0, miny=13.0, maxx=-58.0, maxy=14.0)
