# geoglue merge module
# Merges multiple variables into one dataset
# and then concatenates along the time dimension by default

import shlex
from fileinput import FileInput
from collections import OrderedDict

import xarray as xr


def variable_merge(files: list[str]) -> xr.Dataset:
    to_merge = []
    for file in files:
        ds = xr.open_dataset(file)
        if len(ds.data_vars) == 1:
            v = list(ds.data_vars)[0]
            to_merge.append(ds[v])
        else:
            to_merge.append(ds)
    return xr.merge(to_merge)


def combine_attrs(attrs_list, context):
    """
    attrs_list: sequence of dict-like .attrs from input datasets/arrays
    context: xarray combine context (not used here, but provided by xarray)
    Return: dict of combined attrs
    """
    dicts = [d if d is not None else {} for d in attrs_list]

    # collect ordered set of keys
    keys = OrderedDict()
    for d in dicts:
        for k in d.keys():
            keys.setdefault(k, True)

    out = {}
    for key in keys:
        # collect non-None values in original order
        vals = [d[key] for d in dicts if key in d and d[key] is not None]

        if not vals:
            continue

        if key == "geoglue_config":
            # join unique values while preserving order
            seen = set()
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


def merge_datasets(file_input: FileInput, dim: str = "time") -> xr.Dataset:
    with file_input as data:
        line = next(data)
        ds = variable_merge(shlex.split(line))
        for line in data:
            ds = xr.concat(
                [ds, variable_merge(shlex.split(line))],
                dim=dim,
                combine_attrs=combine_attrs,
            )
    return ds
