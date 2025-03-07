import pytest

from geoglue.util import zero_padded_intrange, is_lonlat


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
