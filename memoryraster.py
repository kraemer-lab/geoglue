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
import copy
import dataclasses
from pathlib import Path

import affine
import rasterio
import geopandas as gpd
import numpy as np
import rasterio.mask
import rasterio.enums
import rasterio.plot
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject


DEFAULT_COLORMAP = "GnBu"


def get_numpy_dtype(t: str):
    match t:
        case "float64":
            return np.float64
        case "float32":
            return np.float32
        case "int32":
            return np.int32
        case "uint32":
            return np.uint32
        case _:
            return np.float64


@dataclasses.dataclass
class MemoryRaster:
    data: np.ndarray | np.ma.MaskedArray
    transform: affine.Affine
    crs: str | None
    nodata: int | float
    origin_path: Path | None = None
    dtype: str = "float64"

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
        }

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
    def read(
        file: str | Path,
        crs: str | None = None,
        resampling: rasterio.enums.Resampling = rasterio.enums.Resampling.bilinear,
    ):
        "Reads from a file supported by rasterio"

        with rasterio.open(file) as src:
            if src.count != 1:
                raise ValueError("Only single band files are supported")
            origin_path = Path(file)
            if crs is None or src.crs == crs:
                # No transformation requested or requested CRS is same as current CRS
                return MemoryRaster(
                    src.read(1), src.transform, src.crs, src.nodata, origin_path
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
            return MemoryRaster(
                reprojected_data.squeeze(), transform, crs, src.nodata, origin_path
            )

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
            with memfile.open(driver="GTiff", **self.profile) as src:
                src.write(self.data, 1)
            with memfile.open() as src:
                masked, transform = rasterio.mask.mask(src, geometry, crop=crop)
                return MemoryRaster(
                    np.ma.masked_equal(masked.squeeze(), self.nodata),
                    transform,
                    self.crs,
                    0,
                )

    def plot(self, cmap: str = DEFAULT_COLORMAP, fill_nodata=0, **kwargs):
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
        print("Resampling using:", resampling)
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

    def zonal_stats(self, geometry: gpd.GeoDataFrame, **kwargs):
        "Calculate zonal statistics using exactextract"
