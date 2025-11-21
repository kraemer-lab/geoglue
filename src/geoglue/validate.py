import itertools

import xarray as xr


def stats_nonregion(da: xr.DataArray):
    """
    Prints statistics for each combination of non-region indices
    """
    isnull = da.isnull().sum().item()
    if isnull:
        print(f"[!!] {isnull} NaN values detected")
    else:
        print("[OK] No NaN values detected")
    non_region_dims = [dim for dim in da.dims if dim != "region"]
    dim_index_ranges = {dim: range(da.sizes[dim]) for dim in non_region_dims}

    # Iterate over all combinations
    for combo in itertools.product(*dim_index_ranges.values()):
        indexer = dict(zip(non_region_dims, combo))
        indexer_s = " ".join(f"{k}={v}" for k, v in indexer.items())

        subset = da.isel(**indexer)  # type: ignore
        mean = subset.mean(dim="region").item()
        minimum = subset.min(dim="region").item()
        maximum = subset.max(dim="region").item()
        print(f"{indexer_s}\tmean={mean:.4f}\tmin={minimum:.4f}\tmax={maximum:.4f}")
