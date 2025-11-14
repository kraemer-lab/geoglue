import os
import textwrap
import pytest
from pathlib import Path

from geoglue.zonalstats_config import ZonalStatsConfig


def write_file(path: Path, content: str):
    path.write_text(textwrap.dedent(content))


def test_read_basic_config_and_instantiate(tmp_path):
    """Simple config with templates and ~ expansion via instantiate."""
    cfg_file = tmp_path / "config.toml"
    # include a variable and templates
    toml = r"""
    raster = "data-$year.nc"
    shapefile = "~/shp/country-$year.shp"
    output = "out-$year.zonal.nc"
    """
    write_file(cfg_file, toml)

    cfg = ZonalStatsConfig.read_file(cfg_file)
    # not instantiated yet: templates remain untouched
    assert "$year" in cfg.raster.template

    instantiated = cfg.instantiate(year=2025)
    assert instantiated.raster == "data-2025.nc"
    # ~ is expanded to actual user home; ensure it starts with HOME
    assert instantiated.shapefile.startswith(os.path.expanduser("~"))
    assert instantiated.shapefile.endswith("country-2025.shp")
    assert instantiated.output == "out-2025.zonal.nc"


def test_includes_merge_and_variable_defaults(tmp_path):
    """Test include merging, variable defaults (weighted_mean when weights present)."""
    inc = tmp_path / "base.toml"
    main = tmp_path / "main.toml"

    write_file(
        inc,
        """
    raster = "base-$year.nc"
    shapefile = "base.shp"
    weights = "weights-$year.tif"

    [variables.t1]
    min = 0
    max = 10
    """,
    )

    write_file(
        main,
        """
    include = ["base.toml"]
    output = "final-$year.nc"

    [variables.t1]
    # override only max
    max = 20

    [variables.t2]
    min = 200
    max = 350
    """,
    )
    cfg = ZonalStatsConfig.read_file(main)
    # include list is present
    assert "base.toml" in cfg.include
    # weights presence should set default operation to weighted_mean for variables without explicit operation
    assert cfg.weights is not None
    assert "t1" in cfg.variables and "t2" in cfg.variables
    assert cfg.variables["t1"].operation == "weighted_mean"
    assert cfg.variables["t1"].min == 0
    assert cfg.variables["t1"].max == 20  # overridden in main.toml
    assert cfg.variables["t2"].min == 200
    assert cfg.variables["t2"].max == 350


def test_template_formats_mixed_and_safe_substitute(tmp_path):
    cfg_file = tmp_path / "cfg.toml"
    write_file(
        cfg_file,
        """
    raster = "r-$region-$year.nc"
    shapefile = "~/shp/$region.shp"
    output = "o-$region-$year.nc"
    """,
    )
    cfg = ZonalStatsConfig.read_file(cfg_file)
    inst = cfg.instantiate(region="north", year=2030)
    assert inst.raster == "r-north-2030.nc"
    assert inst.output == "o-north-2030.nc"
    assert inst.shapefile.endswith("north.shp")


def test_missing_required_fields_raises(tmp_path):
    """Missing raster/shapefile/output should raise on validate/read."""
    cfg_file = tmp_path / "bad.toml"
    # omit shapefile
    write_file(
        cfg_file,
        """
    raster = "something.nc"
    # shapefile missing
    output = "out.nc"
    """,
    )
    with pytest.raises(ValueError):
        ZonalStatsConfig.read_file(cfg_file)


def test_invalid_resample_raises(tmp_path):
    """Top-level invalid resample should raise a ValueError."""
    cfg_file = tmp_path / "bad_resample.toml"
    write_file(
        cfg_file,
        """
    raster = "a.nc"
    shapefile = "s.shp"
    output = "o.nc"
    resample = "not_a_real_resample"
    """,
    )
    with pytest.raises(ValueError):
        ZonalStatsConfig.read_file(cfg_file)


def test_variable_min_greater_than_max_raises(tmp_path):
    cfg_file = tmp_path / "bad_var_bounds.toml"
    write_file(
        cfg_file,
        """
    raster = "a.nc"
    shapefile = "s.shp"
    output = "o.nc"
    [variables.v]
    min = 100
    max = 10
    """,
    )
    with pytest.raises(ValueError):
        ZonalStatsConfig.read_file(cfg_file)


def test_include_cycle_detection(tmp_path):
    """Files that include each other should trigger a cycle error."""
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    write_file(
        a, 'include = ["b.toml"]\nraster="a.nc"\nshapefile="a.shp"\noutput="a.nc"'
    )
    write_file(
        b, 'include = ["a.toml"]\nraster="b.nc"\nshapefile="b.shp"\noutput="b.nc"'
    )
    with pytest.raises(ValueError):
        ZonalStatsConfig.read_file(a)


def test_instantiate_missing_template_key_raises_keyerror(tmp_path):
    """If Python format placeholders are present but missing keys are supplied, expect a KeyError."""
    cfg_file = tmp_path / "cfg2.toml"
    write_file(
        cfg_file,
        """
    raster = "data-$year-$region.nc"
    shapefile = "s.shp"
    output = "o-$year-$region.nc"
    """,
    )
    cfg = ZonalStatsConfig.read_file(cfg_file)
    # only supply year, but region placeholder missing -> KeyError expected
    with pytest.raises(KeyError):
        cfg.instantiate(year=2025)


def test_variables_not_in_output_raises_valueerror(tmp_path):
    cfg_file = tmp_path / "cfg_missing_var_output.toml"
    write_file(
        cfg_file,
        """
    raster = "data-$year-$region.nc"
    shapefile = "s.shp"
    output = "o-$year.nc"
""",
    )
    with pytest.raises(ValueError):
        ZonalStatsConfig.read_file(cfg_file)


def test_template_vars_read(tmp_path):
    cfg_file = tmp_path / "cfg_missing_var_output.toml"
    write_file(
        cfg_file,
        """
    raster = "data-$year-$region.nc"
    shapefile = "s.shp"
    output = "o-$year-$region-$var.nc"
""",
    )
    cfg = ZonalStatsConfig.read_file(cfg_file)
    assert cfg.template_vars == {"year", "region", "var"}


def test_variables_resample_validation(tmp_path):
    """If a variable has an invalid resample operation, should raise during read/validate."""
    cfg_file = tmp_path / "var_resample.toml"
    write_file(
        cfg_file,
        """
    raster = "r.nc"
    shapefile = "s.shp"
    output = "o.nc"
    [variables.v]
    resample = "invalid_op"
    """,
    )
    with pytest.raises(ValueError):
        ZonalStatsConfig.read_file(cfg_file)
