"""
ERA5 data download and data aggregation
---------------------------------------

This module downloads ERA5 daily level data and provides classes to
resample and aggregate data to administrative levels obtained from GADM
"""

from __future__ import annotations

import re

import xarray as xr

from .country import Country
from .zonal_stats import DatasetZonalStatistics

def infer_statistic(ds: xr.Dataset) -> str | None:
    "Infer temporal aggregation statistic from ERA5 NetCDF file"

    # daily_reduce functions by geoglue add a gg_statistic variable
    if "gg_statistic" in ds.attrs:
        return ds.attrs["gg_statistic"]
    last_line_history = ds.attrs["history"].split("\n")[-1]
    # Last line in the history attribute is of this form:
    # earthkit.transforms.aggregate.temporal.daily_reduce(2m_temperature_stream-oper, how=mean, **{'time_shift': {'hours': 7}, 'remove_partial_periods': True})
    # refers to the parameters passed to the daily_reduce function:
    # https://earthkit-transforms.readthedocs.io/en/latest/_api/transforms/aggregate/temporal/index.html#transforms.aggregate.temporal.daily_reduce
    if (m := re.match(r".*how=(\w+)", last_line_history)) and "daily_reduce" in m[0]:
        return "daily_" + m[1]
    return None
