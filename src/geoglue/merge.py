# pyright: reportUnknownMemberType=none, reportUnknownArgumentType=none, reportExplicitAny=none
# geoglue merge module
# Merges multiple variables into one dataset
# and then concatenates along the time dimension by default

from pathlib import Path
from typing import Any
from collections import OrderedDict, defaultdict
from collections.abc import Iterable

import xarray as xr


def variable_merge(files: list[Path]) -> xr.Dataset:
    return xr.merge(
        [xr.open_dataset(f) for f in files],
        combine_attrs=combine_attrs,
    )


def combine_attrs(
    attrs_list: Iterable[dict[str, str | None] | None], context=None
) -> dict[str, str]:  # pyright: ignore[reportUnusedParameter,reportMissingParameterType,reportUnknownParameterType]
    """
    attrs_list: sequence of dict-like .attrs from input datasets/arrays
    context: xarray combine context (not used here, but provided by xarray)
    Return: dict of combined attrs
    """
    dicts: list[dict[str, str | None]] = [
        d if d is not None else {} for d in attrs_list
    ]

    # collect ordered set of keys
    keys: OrderedDict[str, bool] = OrderedDict()
    for d in dicts:
        for k in d.keys():
            keys.setdefault(k, True)  # pyright: ignore[reportUnusedCallResult]

    out: dict[str, str] = {}
    for key in keys:
        # collect non-None values in original order
        vals: list[str] = [d[key] for d in dicts if key in d and d[key] is not None]  # pyright: ignore[reportAssignmentType]

        if not vals:
            continue

        if key == "geoglue_config":
            # join unique values while preserving order
            seen: set[str] = set()
            ordered_unique = []
            for v in vals:
                # if v is bytes, convert to str; otherwise keep as-is
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                if v not in seen:
                    seen.add(v)
                    ordered_unique.append(str(v))
            out[key] = "\n".join(ordered_unique)
        else:
            # keep the first value
            out[key] = vals[0]

    return out


def _group_datasets(files: Iterable[Path], dim: str) -> list[list[Path]]:
    """Groups datasets represented by files by ``dim``.

    Given multiple files, this function groups them into variables that must be
    packed into the same xr.Dataset, sharing the ``dim`` axis. It also orders
    the grouped datasets by ``dim``, so that the datasets can be concatenated.
    """
    groups: defaultdict[tuple[Any, Any], list[Path]] = defaultdict(list)
    vars_in_group: defaultdict[tuple[Any, Any], set[str]] = defaultdict(set)
    diff: Any = None
    for file in files:
        ds = xr.open_dataset(file)
        dim_vals = ds[dim][0].item(), ds[dim][-1].item()  # pyright: ignore[reportAny]
        groups[dim_vals].append(file)
        vars_in_group[dim_vals] |= set(ds.data_vars)
        if diff is None and ds[dim].size > 1:
            diff = ds[dim][1].item() - ds[dim][0].item()  # pyright: ignore[reportAny]
    sorted_dims: list[tuple[Any, Any]] = sorted(vars_in_group.keys())
    first_group_vars = vars_in_group[sorted_dims[0]]

    # check same variable set in each group
    for _, vars in vars_in_group.items():
        if vars != first_group_vars:
            raise ValueError(f"Variable sets in all axis={dim!r} must be identical")

    # check contiguous
    if diff:
        # Example of sorted_dims: [(0, 1), (2, 3), (4, 5)] contiguous, diff=1
        subseq_diffs = [
            fst[0] - snd[1] for fst, snd in zip(sorted_dims[1:], sorted_dims)
        ]
        if subseq_diffs:
            sd0 = subseq_diffs[0]  # pyright: ignore[reportAny]
            if any(sd0 != d for d in subseq_diffs[1:]):  # pyright: ignore[reportAny]
                raise ValueError(f"Concatenation axis {dim!r} not contiguous")
    return [groups[dim_extents] for dim_extents in sorted_dims]


def merge_datasets(files: Iterable[Path], dim: str = "time") -> xr.Dataset:
    file_groups = _group_datasets(files, dim)
    ds = variable_merge(file_groups[0])
    for file_group in file_groups[1:]:
        ds = xr.concat(
            [ds, variable_merge(file_group)],
            dim=dim,
            combine_attrs=combine_attrs,
        )
    return ds
