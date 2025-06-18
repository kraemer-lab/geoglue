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
import shapely.geometry
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject

from .types import CdoGriddes, Bbox
from .util import sha256

DEFAULT_COLORMAP = "viridis"


def grid_size(da: xr.DataArray, axis: str) -> float:
    return abs(float(da[axis][1] - da[axis][0]))


def get_numpy_dtype(t: str):
    if t in ["float64", "float32", "int32", "uint32"]:
        return getattr(np, t)
    return np.float64


@dataclasses.dataclass
class MemoryRaster:
    """Class to operate on rasters in-memory.

    While MemoryRaster can be constructed directly by passing the parameters
    below, in normal practice, it is constructed by reading from a GeoTIFF file
    or xarray object, using :meth:`read()` or :meth:`from_xarray()`.

    Parameters
    ----------
    data : np.ndarray
        Data to consider as raster
    transform : affine.Affine
        Affine transformation associated with raster
    crs : str | pyproj.crs.CRS | None
        Coordinate reference system associated with raster
    nodata : int | float
        Data value indicating NA
    origin_path : Path | None
        Path to source file, optional, default=None.
        This attribute is populated if the MemoryRaster is read from a file
    dtype : str
        numpy dtype of array, default='float64'
    driver : str
        rasterio driver, optional, default='GTiff'
    """

    data: np.ndarray | np.ma.MaskedArray | xr.DataArray
    transform: affine.Affine
    crs: str | pyproj.crs.CRS | None
    nodata: int | float
    origin_path: Path | None = None
    dtype: str = "float64"
    driver: str = "GTiff"

    @property
    def is_lonlat(self):
        "Returns whether grid is longitude and latitude"
        return self.crs is not None and (
            (isinstance(self.crs, pyproj.crs.CRS) and self.crs.to_epsg() == 4326)
            or "4326" in str(self.crs)
        )

    @property
    def shape(self):
        "Shape (width, height) of raster image"
        return self.data.shape  # type: ignore

    @property
    def width(self):
        "Width of raster image"
        return self.shape[1]

    @property
    def height(self):
        "Height of raster image"
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
        "Returns grid description that can be used by cdo to resample"
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

    @property
    def bbox(self) -> Bbox:
        "Bounding box of MemoryRaster"
        g = self.griddes
        return Bbox(
            g.xfirst, g.yfirst, g.xfirst + g.xinc * g.xsize, g.yfirst + g.yinc * g.ysize
        )

    def min(self) -> float:
        "Minimum value in raster"
        return np.ma.min(self.data)

    def max(self) -> float:
        "Maximum value in raster"
        return np.ma.max(self.data)

    def sum(self) -> float:
        "Sum of non-null values in raster"
        return np.ma.sum(self.data)

    def __repr__(self):
        return (
            f"<MemoryRaster {self.shape} CRS={self.crs} "
            f"min={self.data.min()} max={self.data.max()} "
            f"NODATA={self.nodata} file={self.origin_path!r}\n"
            f"  transform={self.transform}>"
        )

    def checksum(self) -> str:
        h = (
            sha256(self.origin_path, prefix=True)
            if self.origin_path
            else "<hash:unknown>"
        )
        return f"MemoryRaster.origin_path={h} {self.origin_path} {self.width}x{self.height}"

    @staticmethod
    def from_xarray(
        da: xr.DataArray,
        c_longitude="longitude",
        c_latitude="latitude",
        nodata: int | float | None = None,
    ) -> MemoryRaster:
        """Creates MemoryRaster from xarray, assumes EPSG:4326

        Parameters
        ----------
        da
            xarray DataArray from which to create MemoryRaster
        c_longitude
            Longitude axis in dataarray, default='longitude'
        c_latitude
            Latitude axis in dataarray, default='latitude'
        nodata
            Data value representing NA, optional. If not specified, tries to read
            from xarray attributes such as `GRIB_missingValue`, `nodata`, `_FillValue`

        Returns
        -------
        MemoryRaster
        """
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
        """Reads from a file supported by rasterio

        Parameters
        ----------
        file
            File to read from, must be openable by rasterio
        crs
            Coordinate reference system to project to
        resampling
            If reprojecting to another CRS, resampling strategy to use. Must
            be a strategy supported by rasterio


        Returns
        -------
        MemoryRaster
        """

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
        "Returns MemoryRaster as a rasterio dataset"
        with MemoryFile() as memfile:
            if zfill and np.ma.isMaskedArray(self.data):
                data = self.data.filled(0)  # type: ignore
                profile = copy.deepcopy(self.profile)
                # zfill is set, so we explicitly ignore nodata
                del profile["nodata"]
            else:
                data = self.data
                profile = self.profile
            with memfile.open(**profile) as dataset:
                dataset.write(np.expand_dims(data, axis=0))

            with memfile.open() as dataset:
                yield dataset

    def mask(
        self,
        geometry: gpd.GeoDataFrame | gpd.GeoSeries | list[shapely.geometry.Polygon],
        crop: bool = True,
    ) -> MemoryRaster:
        """Mask raster file with a set of geometries

        Parameters
        ----------
        geometry
            GeoDataFrame or GeoSeries
        crop
            Whether to crop the extent to the geometry specified, default=True. This
            is passed directly to rasterio.mask.mask.

        Returns
        -------
        MemoryRaster
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
                    self.origin_path,
                )

    def crop(self, bbox: Bbox) -> MemoryRaster:
        "Crop a MemoryRaster to bounds"
        # Check that bbox to crop to is enclosed within present bbox
        if not bbox < self.bbox:
            raise ValueError(f"crop(): provided {bbox!r} not contained in raster {self.bbox!r}")
        return self.mask([bbox.as_polygon()])

    def plot(self, cmap: str = DEFAULT_COLORMAP, fill_nodata=None, **kwargs):
        "Plots a MemoryRaster using sensible defaults"
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
        "Returns a new MemoryRaster with type cast to t"
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

        This function is meant to be used for resampling MemoryRaster, usually
        those created from GeoTIFF files. For data already in netCDF format, we
        recommend using Climate Data Operator (cdo)'s resampling functions, for
        which we provide a wrapper in :mod:`geoglue.resample`

        Parameters
        ----------
        dst
            Destination MemoryRaster
        resampling
            Resampling method, one of `rasterio.enums.Resampling`

        See Also
        --------
        geoglue.resample
            Resample module with wrappers for cdo resample
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
        """Calculate zonal statistics using exactextract

        Parameters
        ----------
        geometry : gpd.GeoDataFrame
            Geometry dataframe, usually read from a shapefile
        ops : str | list[str] | Callable
            exactextract operation(s) to perform
        weights : MemoryRaster | None
            Optional, if specified uses the supplied raster to perform weighted
            zonal statistics
        **kwargs
            Extra parameters passed directly to exactextract.exact_extract()

        Returns
        -------
        pd.DataFrame | gpd.GeoDataFrame
            A copy of the geometry dataframe with additional column(s) with
            the zonal statistics requested. Each separate zonal statistic is
            given a column in the data
        """
        with self.as_rasterio(zfill=True) as raster:
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
