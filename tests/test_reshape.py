import numpy as np
import numpy.testing as npt
import pandas as pd
import xarray as xr
import pytest
from geoglue.reshape import collapse_step_to_month_dataarray, get_durations


@pytest.fixture(scope="session")
def sample_monthly_forecast():
    times = pd.date_range("2024-01-01", periods=4, freq="MS")  # first of each month
    # step coordinates (float, allow NaN)
    # fmt: off
    steps = np.array([
        29., 30., 31.,
        60., 61.,
        91., 92.,
        121., 122.,
        152., 153.,
        182., 183., 184.
    ])
    # fmt: on
    steps = steps * np.timedelta64(1, "D")
    # Make a data array where for each time only one step is "selected" (others NaN)

    # Here steps is an array of 14 floats, but at each timestep (start of each month)
    # only *one* of the steps is selected, and that will always correspond to an
    # integer lead_month 1..6. So out of the 4x14 array (time, step) exactly 4x6 will be filled
    # Goal is to get this down to a non-sparse matrix

    data_single = np.full((len(times), len(steps)), np.nan)
    filled_coordinates = [
        [2, 3, 5, 7, 9, 11],
        [0, 3, 5, 7, 9, 11],
        [2, 4, 6, 8, 10, 13],
        [1, 4, 5, 8, 10, 12],
    ]
    u = 1
    for i, z in enumerate(filled_coordinates):
        for k in z:
            data_single[i, k] = u
            u += 1
    da = xr.DataArray(data_single, coords={"time": times, "step": steps})
    spatial = xr.DataArray(
        np.array([[0.5, 1], [1.5, 3]]),
        dims=("latitude", "longitude"),
        coords={"latitude": np.array([-1, 1]), "longitude": np.array([-2, 2])},
    )
    return da * spatial  # broadcast to include spatial dims


@pytest.fixture(scope="module")
def dense_forecast(sample_monthly_forecast):
    return collapse_step_to_month_dataarray(sample_monthly_forecast)


def test_collapse_step_to_month(dense_forecast):
    assert set(dense_forecast.coords) == {"time", "latitude", "longitude", "month"}
    dense = dense_forecast.transpose("latitude", "longitude", "time", "month")
    expected = np.array(
        [
            [
                [
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                    [3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                    [6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
                    [9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
                ],
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                    [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                ],
            ],
            [
                [
                    [1.5, 3.0, 4.5, 6.0, 7.5, 9.0],
                    [10.5, 12.0, 13.5, 15.0, 16.5, 18.0],
                    [19.5, 21.0, 22.5, 24.0, 25.5, 27.0],
                    [28.5, 30.0, 31.5, 33.0, 34.5, 36.0],
                ],
                [
                    [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
                    [21.0, 24.0, 27.0, 30.0, 33.0, 36.0],
                    [39.0, 42.0, 45.0, 48.0, 51.0, 54.0],
                    [57.0, 60.0, 63.0, 66.0, 69.0, 72.0],
                ],
            ],
        ]
    )
    npt.assert_array_equal(dense.values, expected)


def test_get_durations(dense_forecast):
    durations = get_durations(dense_forecast)
    expected = np.array(
        [
            [31.0, 29.0, 31.0, 30.0, 31.0, 30.0],
            [29.0, 31.0, 30.0, 31.0, 30.0, 31.0],
            [31.0, 30.0, 31.0, 30.0, 31.0, 31.0],
            [30.0, 31.0, 30.0, 31.0, 31.0, 30.0],
        ]
    )
    npt.assert_array_equal(durations, expected)
