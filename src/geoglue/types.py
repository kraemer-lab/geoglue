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
import shapely.geometry
from pyproj import Geod

from cdo import Cdo

geod = Geod(ellps="WGS84")

class Bbox(NamedTuple):
    "Geographic bounding box"

    minx: int | float
    "Western bounds, minimum longitude"
    miny: int | float
    "Southern bounds, minimum latitude"
    maxx: int | float
    "Eastern bounds, maximum longitude"
    maxy: int | float
    "Northern bounds, maximum latitude"

    def __le__(self, other):
        return (
            self.maxy <= other.maxy
            and self.minx >= other.minx
            and self.miny >= other.miny
            and self.maxx <= other.maxx
        )

    def __lt__(self, other):
        return self <= other and not (self == other)

    def __ge__(self, other):
        return (
            self.maxy >= other.maxy
            and self.minx <= other.minx
            and self.miny <= other.miny
            and self.maxx >= other.maxx
        )

    def __gt__(self, other):
        return self >= other and not (self == other)

    def int(self) -> Bbox:
        maxy = self.maxy if isinstance(self.maxy, int) else math.ceil(self.maxy)
        minx = self.minx if isinstance(self.minx, int) else math.floor(self.minx)
        miny = self.miny if isinstance(self.miny, int) else math.floor(self.miny)
        maxx = self.maxx if isinstance(self.maxx, int) else math.ceil(self.maxx)
        return Bbox(minx, miny, maxx, maxy)

    def as_polygon(self) -> shapely.geometry.Polygon:
        return shapely.geometry.box(self.minx, self.miny, self.maxx, self.maxy)

    @property
    def geodetic_area_km2(self) -> float:
        lons = [self.minx, self.maxx, self.maxx, self.minx, self.minx]
        lats = [self.miny, self.miny, self.maxy, self.maxy, self.miny]
    
        area, _ = geod.polygon_area_perimeter(lons, lats)
        return abs(area) / 1e6

    def __str__(self) -> str:
        return f"{self.minx},{self.miny},{self.maxx},{self.maxy}"

    @staticmethod
    def from_xarray(da: xr.DataArray | xr.Dataset) -> Bbox:
        coords = set(da.coords)
        c_lon, c_lat = None, None
        if {"lon", "lat"} < coords:
            c_lon, c_lat = "lon", "lat"
        if {"longitude", "latitude"} < coords:
            c_lon, c_lat = "longitude", "latitude"
        if c_lon is None and c_lat is None:
            raise ValueError("""Can only convert from DataArray or Dataset that has
latitude and longitude information, that has to be stored in
coordinates named lat, lon or latitude, longitude""")
        return Bbox(
            da[c_lon].min().item(),
            da[c_lat].min().item(),
            da[c_lon].max().item(),
            da[c_lat].max().item(),
        )

    @staticmethod
    def from_string(s: str) -> Bbox:
        "Returns Bbox from standard string representation"
        values = [x.strip() for x in s.split(",")]

        def to_num(x):
            return float(x) if "." in x else int(x)

        maxy = to_num(values.pop())
        maxx = to_num(values.pop())
        miny = to_num(values.pop())
        minx = to_num(values.pop())
        return Bbox(minx, miny, maxx, maxy)

    def to_list(self, spec: str) -> list[int | float]:
        """Returns Bbox converted to list of numbers in different order

        The default and standard bbox order is minx,miny,maxx,maxy. Certain
        applications expect the bbox coordinates in a different order. This method
        takes a fmt string and returns a list in that order

        Parameters
        ----------
        spec : str
            Either a fully specified string like "maxx,minx,maxy,maxy" or a
            shorthand. Supported shorthands are "cdsapi" for supplying bbox
            parameters to ECMWF's cdsapi

        Returns
        -------
        list[int | float]
            Returns a list of bbox coordinates in specified order
        """
        if spec == "cdsapi":
            spec = "maxy,minx,miny,maxx"
        specs = [s.strip() for s in spec.split(",")]
        if len(specs) != 4 or set(specs) != {"minx", "miny", "maxx", "maxy"}:
            raise ValueError(f"All bbox coordinates must be specified, got {spec=}")
        bbox_values = {
            "minx": self.minx,
            "miny": self.miny,
            "maxx": self.maxx,
            "maxy": self.maxy,
        }
        return [bbox_values[s] for s in specs]


@dataclass
class CdoGriddes:
    """Grid specification used by Climate Data Operators (CDO)

    This class represents a grid description as specified by the
    Climate Data Operators (cdo) program, with functionality to read
    and write grid descriptions from files.
    """

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
