import numpy as np
import xarray as xr
from numpy.testing import assert_array_equal


def test_apply_ufunc():
    data = np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4)
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1, 2], "lat": [0, 1, 2, 3], "lon": [0, 1, 2, 3]},
        name="cube4",
    )

    def latlon_to_regions(slice2d):
        """
        slice2d is a 2D numpy array shaped (4,4).
        Return array of length 4 with means of the 2x2 blocks:
         [top-left, top-right, bottom-left, bottom-right]
        """
        a = np.asarray(slice2d)
        r0 = a[0:2, 0:2].mean()  # top-left
        r1 = a[0:2, 2:4].mean()  # top-right
        r2 = a[2:4, 0:2].mean()  # bottom-left
        r3 = a[2:4, 2:4].mean()  # bottom-right
        return np.array([r0, r1, r2, r3], dtype=a.dtype)

    out = xr.apply_ufunc(
        latlon_to_regions,
        da,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["region"]],
        vectorize=True,  # apply to each time slice
    )

    out = out.assign_coords(region=("region", ["r0", "r1", "r2", "r3"]))

    # Compute block means directly
    expected_vals = []
    for t in range(data.shape[0]):
        sl = data[t]
        e0 = sl[0:2, 0:2].mean()
        e1 = sl[0:2, 2:4].mean()
        e2 = sl[2:4, 0:2].mean()
        e3 = sl[2:4, 2:4].mean()
        expected_vals.append([e0, e1, e2, e3])
    expected = xr.DataArray(
        np.array(expected_vals, dtype=float),
        dims=("time", "region"),
        coords={"time": [0, 1, 2], "region": ["r0", "r1", "r2", "r3"]},
        name="expected_regions",
    )

    assert_array_equal(out.values, expected.values)
