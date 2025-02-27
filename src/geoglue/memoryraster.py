"""
MemoryRaster class to read and operate on raster files with metadata entirely
in memory.

While rasterio offers low level access to read and manipulate geospatial raster
files (such as in GeoTIFF), it does not have easy to use higher level functions
for standard operations on rasters such as projection to a different coordinate
system, resampling, or zonal statistics. This module defines a MemoryRaster
class to contain metadata about rasters, to and fro conversion from
rasterio.DataReader, and functions for plotting, reprojection and resampling.
These functions are intended to make working with rasters in Python as easy to
use as R's terra package.
"""

from __future__ import annotations

import pyproj
import xarray as xr
import copy
import dataclasses
from typing import Callable
from pathlib import Path
from contextlib import contextmanager

import affine
import exactextract
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import rasterio.mask
import rasterio.plot

from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject

from .types import CdoGriddes

DEFAULT_COLORMAP = "viridis"


def grid_size(da: xr.DataArray, axis: str) -> float:
    return abs(float(da[axis][1] - da[axis][0]))


def get_numpy_dtype(t: str):
    if t in ["float64", "float32", "int32", "uint32"]:
        return getattr(np, t)
    return np.float64


@dataclasses.dataclass
class MemoryRaster:
    data: np.ndarray | np.ma.MaskedArray | xr.DataArray
    transform: affine.Affine
    crs: str | pyproj.crs.CRS | None
    nodata: int | float
    origin_path: Path | None = None
    dtype: str = "float64"
    driver: str = "GTiff"

    @property
    def is_lonlat(self):
        return self.crs is not None and (
            (isinstance(self.crs, pyproj.crs.CRS) and self.crs.to_epsg() == 4326)
            or "4326" in str(self.crs)
        )

    @property
    def shape(self):
        return self.data.shape  # type: ignore

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    @property
    def profile(self):
        return {
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "count": 1,
            "dtype": self.dtype,
            "driver": self.driver,
            "nodata": self.nodata,
        }

    @property
    def griddes(self) -> CdoGriddes:
        if not self.is_lonlat:
            raise ValueError(
                "Only EPSG:4326 (WGS84) CRS latitude/longitude is supported.\n"
                "Reproject CRS by passing crs='EPSG:4326' to MemoryRaster"
            )
        gridtype = "lonlat"
        xunits = "degrees_east"
        xname = xlongname = "longitude"
        yname = ylongname = "latitude"
        yunits = "degrees_north"
        ysize, xsize = self.shape
        gridsize = ysize * xsize
        if isinstance(self.data, xr.DataArray):
            xinc = grid_size(self.data, "longitude")
            yinc = grid_size(self.data, "latitude")
            xfirst = float(self.data["longitude"].min())
            yfirst = float(self.data["latitude"].min())
        else:
            xinc = self.transform[0]
            # Latitude is descending, so the latitude increment is negative
            yinc = abs(self.transform[4])
            xfirst = self.transform[2]
            yfirst = self.transform[5] + self.transform[4] * ysize
        return CdoGriddes(
            gridtype=gridtype,
            gridsize=gridsize,
            xsize=xsize,
            ysize=ysize,
            xname=xname,
            xlongname=xlongname,
            xunits=xunits,
            yname=yname,
            ylongname=ylongname,
            yunits=yunits,
            xfirst=xfirst,
            xinc=xinc,
            yfirst=yfirst,
            yinc=yinc,
        )

    def min(self) -> float:
        return np.ma.min(self.data)

    def max(self) -> float:
        return np.ma.max(self.data)

    def sum(self) -> float:
        return np.ma.sum(self.data)

    def __repr__(self):
        return (
            f"<MemoryRaster {self.shape} CRS={self.crs} "
            f"min={self.data.min()} max={self.data.max()} "
            f"NODATA={self.nodata} file={self.origin_path!r}\n"
            f"  transform={self.transform}>"
        )

    @staticmethod
    def from_xarray(
        da: xr.DataArray,
        c_longitude="longitude",
        c_latitude="latitude",
        nodata: int | float | None = None,
    ) -> MemoryRaster:
        "Creates MemoryRaster from xarray, assumes EPSG:4326"
        da = da.sortby(da.latitude, ascending=False)
        attrs = da.attrs

        if nodata is None:
            # When reading netCDF files, nodata value is stored in certain
            # attributes This is the list of attributes to search, in order, to
            # obtain nodata value
            nodata = (
                attrs.get("GRIB_missingValue")
                or attrs.get("nodata")
                or attrs.get("_FillValue")
                or attrs.get("missing_value")
            )

        transform = from_origin(
            da[c_longitude].min(),
            da[c_latitude].max(),
            grid_size(da, c_longitude),
            grid_size(da, c_latitude),
        )
        return MemoryRaster(
            da, transform, "EPSG:4326", nodata=nodata or 0, dtype=da.dtype.name
        )

    @staticmethod
    def read(
        file: str | Path,
        crs: str | None = None,
        resampling: rasterio.enums.Resampling = rasterio.enums.Resampling.bilinear,
    ) -> MemoryRaster:
        "Reads from a file supported by rasterio"

        with rasterio.open(file) as src:
            if src.count != 1:
                raise ValueError("Only single band files are supported")
            origin_path = Path(file)
            if crs is None or src.crs == crs:
                # No transformation requested or requested CRS is same as current CRS
                data = src.read(1)
                return MemoryRaster(
                    np.ma.masked_equal(data, src.nodata),
                    src.transform,
                    src.crs,
                    src.nodata,
                    origin_path,
                )
            transform, width, height = calculate_default_transform(
                src.crs, crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update(
                {"crs": crs, "transform": transform, "width": width, "height": height}
            )
            assert isinstance(width, int) and isinstance(height, int)

            reprojected_data = np.zeros(
                (height, width), dtype=get_numpy_dtype(kwargs["dtype"])
            )
            _, transform = reproject(
                source=rasterio.band(src, 1),
                destination=reprojected_data,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=crs,
                resampling=resampling,
            )
            data = reprojected_data.squeeze()
            return MemoryRaster(
                np.ma.masked_equal(data, src.nodata),
                transform,
                crs,
                src.nodata,
                origin_path,
            )

    @contextmanager
    def as_rasterio(self, zfill: bool = False):
        with MemoryFile() as memfile:
            if zfill and np.ma.isMaskedArray(self.data):
                data = self.data.filled(0)  # type: ignore
            else:
                data = self.data
            with memfile.open(**self.profile) as dataset:
                dataset.write(np.expand_dims(data, axis=0))

            with memfile.open() as dataset:
                yield dataset

    def mask(
        self, geometry: gpd.GeoDataFrame | gpd.GeoSeries, crop: bool = True
    ) -> MemoryRaster:
        """Mask raster file with a set of geometries

        Parameters
        ----------
        geometry
            GeoDataFrame or GeoSeries
        crop
            Whether to crop the extent to the geometry specified, default=True. This
            is passed directly to rasterio.mask.mask.
        """
        if isinstance(geometry, gpd.GeoDataFrame):
            # reassign geometry to GeoSeries
            geometry = geometry.geometry
        with MemoryFile() as memfile:
            with memfile.open(**self.profile) as src:
                src.write(self.data, 1)
            with memfile.open() as src:
                masked, transform = rasterio.mask.mask(src, geometry, crop=crop)
                return MemoryRaster(
                    np.ma.masked_equal(masked.squeeze(), self.nodata),
                    transform,
                    self.crs,
                    self.nodata,
                )

    def plot(self, cmap: str = DEFAULT_COLORMAP, fill_nodata=None, **kwargs):
        if fill_nodata is None:
            return rasterio.plot.show(
                self.data, transform=self.transform, cmap=cmap, **kwargs
            )
        else:
            return rasterio.plot.show(
                np.ma.filled(self.data, fill_nodata),
                transform=self.transform,
                cmap=cmap,
                **kwargs,
            )

    def astype(self, t) -> MemoryRaster:
        out = copy.deepcopy(self)
        out.data = out.data.astype(t)
        out.dtype = t.__name__
        return out

    def resample(
        self,
        dst: MemoryRaster,
        resampling: rasterio.enums.Resampling,
    ) -> MemoryRaster:
        """Resamples source raster to match destination mask

        NOTE: This function is not working at the moment

        Parameters
        ----------
        dst
            Destination MemoryRaster
        resampling
            Resampling method, one of `rasterio.enums.Resampling`
        """
        # drop the first index (1)
        height, width = dst.shape  # type: ignore
        assert isinstance(width, int) and isinstance(height, int)
        # Create an empty array to hold the resampled data
        resampled_data = np.zeros((height, width), dtype=get_numpy_dtype(self.dtype))

        reproject(
            source=np.ma.filled(
                self.data, 0
            ),  # replace nodata with 0 before resampling
            destination=resampled_data,
            src_transform=self.transform,
            src_crs=self.crs,
            dst_transform=dst.transform,
            dst_crs=dst.crs,
            resampling=resampling,  # Choose resampling method (nearest, bilinear, etc.)
        )
        return MemoryRaster(
            np.ma.masked_equal(resampled_data, self.nodata), dst.transform, dst.crs, 0
        )

    def zonal_stats(
        self,
        geometry: gpd.GeoDataFrame,
        ops: str | list[str] | Callable,
        weights: MemoryRaster | None = None,
        **kwargs,
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        "Calculate zonal statistics using exactextract"

        with self.as_rasterio() as raster:
            if weights:
                with weights.as_rasterio(zfill=True) as weights_raster:
                    out = exactextract.exact_extract(
                        raster,
                        geometry,
                        ops,
                        weights=weights_raster,
                        output="pandas",
                        **kwargs,
                    )
            else:
                out = exactextract.exact_extract(
                    raster, geometry, ops, output="pandas", **kwargs
                )
        if not isinstance(out, (pd.DataFrame, gpd.GeoDataFrame)):
            raise ValueError(
                f"exactextract failed to calculate zonal statistics, returned output:\n{out}"
            )
        return out
