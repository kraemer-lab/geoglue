import os
import pytest
from pathlib import Path
from geoglue.config import CropConfig, CropConfigTemplate, ZonalStatsTemplate
from geoglue.types import Bbox


@pytest.fixture(scope="session")
def crop_config():
    return CropConfigTemplate.from_dict(
        {
            "raster": "WLD-$year-dataset.nc",
            "region": "data/VNM/geoboundaries/geoBoundaries-VNM-ADM1.shp",
            "integer_bounds": True,
            "split": False,
            "output": "output/VNM-1-$year-dataset.nc",
        }
    )


def test_fill_crop_config(crop_config):
    cfg = crop_config.fill(year=2015)
    assert cfg == CropConfig(
        Path("WLD-2015-dataset.nc"),
        Bbox(minx=102, miny=8, maxx=115, maxy=24),
        Path("output/VNM-1-2015-dataset.nc"),
        split=False,
    )


@pytest.fixture(scope="session")
def unweighted_config():
    return ZonalStatsTemplate.from_dict(
        {
            "raster": "data-$year.nc",
            "shapefile": "~/shp/country-$year.shp",
            "shapefile_id": "id",
            "output": "out-$year.zonal.nc",
            "operation": "mean",
        }
    )


@pytest.fixture(scope="session")
def weighted_config():
    return ZonalStatsTemplate.from_dict(
        {
            "raster": "data $year.nc",
            "shapefile": "shp/country-$year.shp",
            "shapefile_id": "id",
            "output": "output $year.zonal.nc",
            "operation": "mean",
            "weights": "weights $year.tif",
            "resample": "remapbil",
        }
    )


def test_strconfig(weighted_config):
    cfg = weighted_config.fill(year=2015)
    assert str(cfg) == (
        """raster="data 2015.nc" """
        """shapefile=shp/country-2015.shp """
        """shapefile_id=id output="output 2015.zonal.nc" """
        """operation=weighted_mean weights="weights 2015.tif" resample=remapbil"""
    )


def test_read_basic_config_and_instantiate(unweighted_config):
    """Simple config with templates and ~ expansion via instantiate."""

    instantiated = unweighted_config.fill(year=2025)
    assert instantiated.raster == Path("data-2025.nc")
    # ~ is expanded to actual user home; ensure it starts with HOME
    shapefile = str(instantiated.shapefile)
    assert shapefile.startswith(os.path.expanduser("~"))
    assert shapefile.endswith("country-2025.shp")
    assert instantiated.output == Path("out-2025.zonal.nc")


def test_missing_required_fields_raises():
    """Missing raster/shapefile/output should raise on validate/read."""
    with pytest.raises(KeyError):
        ZonalStatsTemplate.from_dict(
            {
                "raster": "something.nc",
                "operation": "mean",
                # shapefile missing
                "output": "out.nc",
            }
        )


def test_invalid_resample_raises():
    """Top-level invalid resample should raise a ValueError."""
    with pytest.raises(ValueError):
        ZonalStatsTemplate.from_dict(
            {
                "raster": "a.nc",
                "operation": "mean",
                "shapefile": "s.shp",
                "shapefile_id": "id",
                "output": "a.nc",
                "resample": "remapnon",
            }
        )


def test_instantiate_missing_template_key_raises_keyerror():
    """If Python format placeholders are present but missing keys are supplied, expect a KeyError."""
    cfg = ZonalStatsTemplate.from_dict(
        {
            "raster": "data-$year-$region.nc",
            "operation": "mean",
            "shapefile": "s.shp",
            "shapefile_id": "id",
            "output": "o-$year-$region.nc",
        }
    )
    # only supply year, but region placeholder missing -> KeyError expected
    with pytest.raises(KeyError):
        cfg.fill(year=2025)


def test_variables_not_in_output_raises_valueerror():
    with pytest.raises(ValueError):
        ZonalStatsTemplate.from_dict(
            {
                "raster": "data-$year-$region.nc",
                "operation": "mean",
                "shapefile": "s.shp",
                "shapefile_id": "id",
                "output": "o-$year.nc",
            }
        )


def test_template_vars_read():
    cfg = ZonalStatsTemplate.from_dict(
        {
            "raster": "data-$year-$region.nc",
            "shapefile": "s.shp",
            "output": "o-$year-$region.nc",
            "shapefile_id": "id",
            "operation": "mean",
        }
    )
    assert cfg.template_vars == {"year", "region"}
