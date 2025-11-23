import os
from pathlib import Path
from geoglue.config import GeoglueConfig, ShapefileConfig, read_config

area_weighted_sum = {
    "area_weighted_sum": "area_weighted_sum(coverage_weight=area_spherical_km2,default_weight=0)"
}
geoglue_config = GeoglueConfig(
    operation=area_weighted_sum,
    region={
        "VNM2": ShapefileConfig(
            Path("data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp"), "shapeID"
        )
    },
    paths={},
    source=None,
)


def test_shapefile_config():
    assert ShapefileConfig.from_str("VNM-2.shp::GID_2") == ShapefileConfig(
        Path("VNM-2.shp"), "GID_2"
    )


def test_geoglue_config():
    assert (
        GeoglueConfig.from_dict(
            {
                "operation": area_weighted_sum,
                "region": {
                    "VNM2": {
                        "file": "data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp",
                        "pk": "shapeID",
                    }
                },
            }
        )
        == geoglue_config
    )


def test_geoglue_config_read_from_file(tmp_path):
    tmp_file = tmp_path / "geoglue_config.toml"
    os.environ["TMP_GEOGLUE_PATH"] = "/scratch/geoglue"
    tmp_file.write_text("""
[paths]
tmp = "$TMP_GEOGLUE_PATH"

[operation]
area_weighted_sum = "area_weighted_sum(coverage_weight=area_spherical_km2,default_weight=0)"

[region.VNM2]
file = "data/VNM/geoboundaries/geoBoundaries-VNM-ADM2.shp"
pk = "shapeID"
""")
    cfg = GeoglueConfig(
        geoglue_config.operation,
        geoglue_config.region,
        {"tmp": Path("/scratch/geoglue")},
        tmp_file,
    )
    assert GeoglueConfig.read_file(tmp_file) == cfg
    assert read_config(tmp_file) == cfg
    del os.environ["TMP_GEOGLUE_PATH"]
