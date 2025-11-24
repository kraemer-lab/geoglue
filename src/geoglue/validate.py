import itertools
from pathlib import Path
from typing import NamedTuple

import xarray as xr

CoordinateValue = dict[str, int]


class VariableStatistics(NamedTuple):
    mean: float
    min: float
    max: float


class Statistics(NamedTuple):
    nna: int
    mean: float
    min: float
    max: float


class CoordStatistics(NamedTuple):
    coord: CoordinateValue
    stat: VariableStatistics

    def __str__(self) -> str:
        _coord_str = " ".join(f"{k}={v}" for k, v in self.coord.items())
        return f"{_coord_str}\tmean={self.stat.mean:.5f}\tmin={self.stat.min:.5f}\tmax={self.stat.max:.5f}"


class DataArrayStatistics(NamedTuple):
    name: str
    nna: int  # number of NA values
    coord_stats: list[CoordStatistics]


def minimal_stats(da: xr.DataArray) -> Statistics:
    return Statistics(
        nna=da.isnull().sum().item(),
        mean=da.mean().item(),
        max=da.max().item(),
        min=da.min().item(),
    )


def stats(
    da: xr.DataArray,
) -> DataArrayStatistics:
    """
    Prints statistics for each combination of non-region indices
    """
    if "region" not in da.dims:
        raise ValueError(
            "Not a zonalstats file (.zs.nc), must have the 'region' coordinate"
        )
    nna = da.isnull().sum().item()
    non_region_dims = [dim for dim in da.dims if dim != "region"]
    dim_index_ranges = {dim: range(da.sizes[dim]) for dim in non_region_dims}

    # Iterate over all combinations
    out = []
    for combo in itertools.product(*dim_index_ranges.values()):
        indexer = dict(zip(map(str, non_region_dims), combo))

        subset = da.isel(**indexer)  # type: ignore
        mean = subset.mean(dim="region").item()
        minimum = subset.min(dim="region").item()
        maximum = subset.max(dim="region").item()
        out.append(
            CoordStatistics(
                indexer, VariableStatistics(mean=mean, min=minimum, max=maximum)
            )
        )
    name = str(da.name) if da.name else "var"
    return DataArrayStatistics(name, nna, out)


def array_size(da: xr.DataArray) -> str:
    return " ".join(f"{k}={v}" for k, v in da.sizes.items())


def print_file_stats(file: Path):
    ds = xr.open_dataset(file)
    for var in sorted(ds.data_vars):
        size_str = array_size(ds[var])
        if "region" in ds[var].dims:
            st = stats(ds[var])
            nna = st.nna
            if nna == 0:
                print(f"\033[1m{file.name}:{var}: nna={nna}\033[0m")
            else:
                print(f"\033[1m{file.name}:{var}: nna=\033[1;91{nna}\033[0m")
            print(f"{file.name}:{var}: \033[0;36m{size_str}\033[0m")
            for coord_stat in st.coord_stats:
                print(f"{file.name}:{var}: {coord_stat}")
        else:
            ms = minimal_stats(ds[var])
            print(f"\033[1m{file.name}:{var}: nna={ms.nna}\033[0m")
            print(f"{file.name}:{var}: \033[0;34m{size_str}\033[0m")
            print(
                f"{file.name}:{var}: mean={ms.mean:.5f}\tmin={ms.min:.5f}\tmax={ms.max:.5f}"
            )
