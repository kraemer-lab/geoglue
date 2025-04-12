import tempfile

import pytest
import xarray as xr
import numpy as np

from geoglue.types import CdoGriddes
from geoglue.util import (
    zero_padded_intrange,
    is_lonlat,
    set_lonlat_attrs,
    find_unique_time_coord,
)


@pytest.mark.parametrize(
    "start,end,inclusive,expected",
    [
        (1, 11, False, ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]),
        (
            1,
            11,
            True,
            ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
        ),
    ],
)
def test_zero_padded_intrange(start, end, inclusive, expected):
    assert zero_padded_intrange(start, end, inclusive) == expected


def test_is_lonlat():
    assert is_lonlat("data/VNM/era5/VNM-2020-era5.daily_sum.nc") is True


def test_set_lonlat_attrs(temp_dataset):
    """Test that set_lonlat_attrs() sets grid attributes correctly
    so that cdo griddes detects it as a 'lonlat' grid"""

    assert CdoGriddes.from_dataset(temp_dataset).gridtype == "generic"
    # fix gridtype by adding coordinate attributes
    set_lonlat_attrs(temp_dataset)
    assert CdoGriddes.from_dataset(temp_dataset).gridtype == "lonlat"


def test_find_unique_time_coords_success(temp_dataset):
    assert find_unique_time_coord(temp_dataset) == "time"


def test_find_unique_time_coords_success_raises_multiple(
    temp_dataset_multiple_time_axes,
):
    with pytest.raises(ValueError, match="No unique time coordinate found"):
        find_unique_time_coord(temp_dataset_multiple_time_axes)


def test_find_unique_time_fails_without_any_coordinate():
    lat = [0, 1, 2]
    lon = [-1, 0, 1]
    data = np.random.rand(len(lat), len(lon))
    ds = xr.Dataset(
        {"temperature": (["latitude", "longitude"], data)},
        coords={"latitude": lat, "longitude": lon},
    )
    with pytest.raises(ValueError, match="No time coordinate found"):
        find_unique_time_coord(ds)
