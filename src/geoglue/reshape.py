import xarray as xr
import numpy as np
import pandas as pd


def collapse_step_to_month_dataarray(
    da: xr.DataArray, time_coord: str = "time"
) -> xr.DataArray:
    """Given a DataArray with (time, step) variables

    For cdsapi seasonal forecast monthly data, data is presented as a (time,
    step) coordinate where time = YYYY-MM-01 (monthly timesteps), but step is
    presented as timesteps in days with type timedelta64[ns]. This results in a
    sparse representation as the step coordinate is all possible values e.g. 29,
    30, 31, 60, 61, 62, 91, 92, 93 while only one of them is selected at each
    timestep so for time=2024-01-01, step=31, 60, 91 is non-nan.

    This function converts (time, step) to a (time, month) dense representation
    with month going from 1..6. As the forecast data always includes a fixed
    number of steps, this dense operation is valid.
    """
    if not {time_coord, "latitude", "longitude", "step"} <= set(da.coords):
        raise ValueError(
            "Invalid DataArray passed, must have (time, step) coordinates and spatial (latitude, longitude) coordinates"
        )

    # TODO: calculate valid_time = time + step
    # TODO: check all values where da != NaN correspond to valid_time YYYY-MM-01

    da = da.transpose("time", "latitude", "longitude", "step")
    arr = da.values  # shape (T, Y, X, S)
    T, Y, X, S = arr.shape
    arr_flat = arr.reshape(T * Y * X, S)

    # collapse non-NaNs per spatial–time point
    dense_flat = np.vstack([row[~np.isnan(row)] for row in arr_flat])

    # infer number of valid steps (assume constant)
    m = dense_flat.shape[1]
    dense = dense_flat.reshape(T, Y, X, m)

    # build new DataArray with 'month' axis
    return xr.DataArray(
        dense,
        dims=("time", "latitude", "longitude", "month"),
        coords={
            time_coord: da[time_coord],
            "latitude": da["latitude"],
            "longitude": da["longitude"],
            "month": np.arange(1, m + 1),
        },
        name=da.name if da.name else None,
    )


def collapse_step_to_month(ds: xr.Dataset, time_coord="time") -> xr.Dataset:
    "Collapses step to month for a Dataset, across all data variables using collapse_step_to_month_dataarray()"
    return xr.Dataset(
        {
            var: collapse_step_to_month_dataarray(ds[var], time_coord)
            for var in ds.data_vars
        }
    )


def days_in_nth_month(t: int, n_month: int) -> int:
    "Returns number of days in the n_month after time"
    time = pd.Timestamp(t)
    if n_month < 1:
        raise ValueError("Negative n_month not supported")
    if n_month > 12:
        raise ValueError("Only supports n_month 1..12")
    if time.day != 1:
        raise ValueError("Can only work with start of months")
    new_month = (time.month + n_month - 1) % 12 + 1
    new_year = time.year + (time.month + n_month) // 12
    prev_month = (new_month - 2) % 12 + 1
    prev_month_year = new_year if prev_month != 12 else new_year - 1
    return (
        pd.Timestamp(new_year, new_month, 1)
        - pd.Timestamp(prev_month_year, prev_month, 1)
    ).days


def get_durations(da: xr.DataArray) -> xr.DataArray:
    """Gets duration in days for (time, month) coordinate

    For a DataArray with (time, month) coordinate, returns a duration_days
    (time, month) DataArray with the duration in days for each monthly offset.

    For example with a time coordinate=2024-01-01 and a month coordinate of
    1..6, will return an array of duration in days for each month offset:
      [31., 29., 31., 30., 31., 30.]
    """
    X = np.zeros((da.time.shape[0], da.month.shape[0]))
    for t_i, t in enumerate(da.time):
        for m_i, m in enumerate(da.month):
            X[t_i, m_i] = days_in_nth_month(t.item(), m.item())
    return xr.DataArray(
        X, dims=("time", "month"), coords={"time": da.time, "month": da.month}
    )
