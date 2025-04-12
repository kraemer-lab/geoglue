"Common types used in geoglue"

from __future__ import annotations

import math
import copy
import tempfile
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass, asdict

import numpy as np
import xarray as xr

from cdo import Cdo


class Bounds(NamedTuple):
    "Geographic bounds"

    north: int | float
    west: int | float
    south: int | float
    east: int | float

    def __lt__(self, other):
        return (
            self.north <= other.north
            and self.west >= other.west
            and self.south >= other.south
            and self.east <= other.east
        )

    def __gt__(self, other):
        return (
            self.north >= other.north
            and self.west <= other.west
            and self.south <= other.south
            and self.east >= other.east
        )

    def integer_bounds(self) -> Bounds:
        north = self.north if isinstance(self.north, int) else math.ceil(self.north)
        west = self.west if isinstance(self.west, int) else math.floor(self.west)
        south = self.south if isinstance(self.south, int) else math.floor(self.south)
        east = self.east if isinstance(self.east, int) else math.ceil(self.east)
        return Bounds(north, west, south, east)


@dataclass
class CdoGriddes:
    "Grid specification"

    gridtype: str
    gridsize: int
    xsize: int
    ysize: int
    xname: str
    yname: str
    xfirst: float
    xinc: float
    yfirst: float
    yinc: float
    ylongname: str = "latitude"
    yunits: str = "degrees_north"
    xlongname: str = "longitude"
    xunits: str = "degrees_east"

    def __str__(self) -> str:
        out = []
        for k, v in asdict(self).items():
            if k.endswith("units") or k.endswith("longname"):
                out.append(f'{k:9s} = "{v}"')
            else:
                out.append(f"{k:9s} = {v}")
        return "\n".join(out)

    @staticmethod
    def from_file(
        file: str | Path, base: CdoGriddes | None = None, **kwargs
    ) -> CdoGriddes:
        out = {}
        _cdo = Cdo()
        for line in _cdo.griddes(input=str(file)):
            if line.startswith("#"):
                continue
            key, value = line.split("=")
            key = key.strip()
            value = value.strip()
            # do not record default scanningMode = 64
            # see https://sources.debian.org/src/cdo/2.5.0-1/src/griddes.h/#L37
            if key == "scanningMode" and int(value) == 64:
                continue
            if key.endswith("size"):
                out[key] = int(value)
                continue
            if key.endswith("first") or key.endswith("inc"):
                out[key] = float(value)
            else:
                out[key] = value.replace('"', "")
        if base:
            new_out = copy.deepcopy(asdict(base))
            # override keys already found with keys in file
            new_out.update(out)
        else:
            new_out = out
        if kwargs:
            new_out.update(kwargs)
        return CdoGriddes(**out)

    @staticmethod
    def from_dataset(ds: xr.Dataset) -> CdoGriddes:
        with tempfile.NamedTemporaryFile(prefix="geoglue-", suffix=".nc") as f:
            ds.to_netcdf(f.name)
            return CdoGriddes.from_file(f.name)
        
    def approx_equal(self, other: CdoGriddes, rtol=1e-05, atol=1e-08) -> bool:
        "Approximate equality testing, with absolute (atol) and relative (rtol) tolerance"
        float_fields = ["xfirst", "yfirst", "xinc", "yinc"]
        this_floats = np.array([getattr(self, f) for f in float_fields])
        other_floats = np.array([getattr(other, f) for f in float_fields])

        this_dict = {k: v for k, v in asdict(self).items() if k not in float_fields}
        other_dict = {k: v for k, v in asdict(self).items() if k not in float_fields}

        return this_dict == other_dict and np.allclose(
            this_floats, other_floats, rtol=rtol, atol=atol
        )

    def write(self, file: str | Path):
        Path(file).write_text(str(self))
