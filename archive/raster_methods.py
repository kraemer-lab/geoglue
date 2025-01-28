import os
import logging
import tempfile
from pathlib import Path

import rasterio
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import rasterio.mask
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

def convert_crs(
    file: str | Path, dst_crs: str = "EPSG:4326", resampling=Resampling.bilinear
) -> str | Path:
    "Reproject raster to different CRS"
    with rasterio.open(file) as src:
        if src.crs == dst_crs:  # no projection needed
            logging.info(
                f"No projection needed as source and destination CRS are identical: {src.crs}"
            )
            return file
        print("Converting CRS for:", file)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        handle, tmpfile = tempfile.mkstemp(suffix=".tif", prefix="dartagg")
        os.close(handle)

        with rasterio.open(tmpfile, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )
        return tmpfile

def raster_mask(file: str | Path, geometry: gpd.GeoSeries) -> npt.ArrayLike:
    """Mask raster file with a set of geometries

    Parameters
    ----------
    file
        Raster file to read
    geometry
        Geometry column of a GeoDataFrame, can be accessed by the .geometry attribute

    Returns
    -------
    np.array
        Array of the same shape as the raster data, but with values outside the polygons
        defined by the geometry set to NaN
    """

    print("\nraster_mask reading:", file)
    with rasterio.open(file) as src:
        masked, transform = rasterio.mask.mask(src, geometry, crop=True)
        masked = masked.squeeze()
        masked[masked == src.nodata] = 0
        print("\tcrs:", src.crs)
        print("\tbounds:", src.bounds)
    print("\ttransform:", transform)
    print("\tshape:", masked.shape)
    return masked, transform

def resample_to_destination(
    source_data: npt.ArrayLike,
    source_transform,
    destination: npt.ArrayLike,
    destination_transform,
    crs: CRS | str,
    resampling,
) -> npt.ArrayLike:
    """Resamples source raster to match destination mask

    NOTE: This function is not working at the moment

    Parameters
    ----------
    source_data
        Source raster file
    source_transform
        Source transform function
    destination
        Destination raster data as a numpy array, this is only used to get the shape
    destination_transform
        Destination transform
    target_crs
        Target CRS
    bounds
         Geometry bounds
    resampling
        Resampling method, one of `rasterio.enums.Resampling`

    Returns
    -------
    np.array
        Resampled raster data in an array
    """
    print("Resampling using:", resampling)
    # drop the first index (1)
    height, width = destination.shape  # type: ignore
    assert isinstance(width, int) and isinstance(height, int)
    # Create an empty array to hold the resampled data
    resampled_data = np.zeros((height, width), dtype=np.float32)

    # Perform the resampling using reproject function
    reproject(
        source=source_data,
        destination=resampled_data,
        src_transform=source_transform,
        src_crs=crs,
        dst_transform=destination_transform,
        dst_crs=crs,
        resampling=resampling,  # Choose resampling method (nearest, bilinear, etc.)
    )
    return resampled_data
