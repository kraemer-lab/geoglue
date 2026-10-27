import os
from pathlib import Path

from geoglue.config import (
    GeoglueConfig,
    ShapefileConfig,
    ZonalStatsConfig,
    read_config,
    read_zonalstats_config,
)

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


zonalstats_config1 = (
    "raster=data/country/BRA-2015-gdp_pc.nc "
    "shapefile=data/geometry/BR_Municipios_2023.shp "
    "shapefile_id=CD_MUN "
    "output=BRA-2015-gdp_pc.zs.nc "
    "operation=weighted_mean(coverage_weight=area_spherical_km2,default_weight=0) "
    "weights=data/worldpop/bra_pop_2015_CN_1km_R2025A_UA_v1.tif "
    "resample=sremapbil "
)


zonalstats_config2 = (
    "raster=data/country/BRA-2016-gdp_pc.nc "
    "shapefile=data/geometry/BR_Municipios_2023.shp "
    "shapefile_id=CD_MUN "
    "output=BRA-2016-gdp_pc.zs.nc "
    "operation=weighted_mean(coverage_weight=area_spherical_km2,default_weight=0) "
    "weights=data/worldpop/bra_pop_2016_CN_1km_R2025A_UA_v1.tif "
    "resample=sremapbil "
)


def test_shapefile_config():
    assert ShapefileConfig.from_str("VNM-2.shp::GID_2") == ShapefileConfig(
        Path("VNM-2.shp"), "GID_2"
    )


def test_zonalstats_config_from_str():
    assert read_zonalstats_config(zonalstats_config1) == ZonalStatsConfig(
        raster=Path("data/country/BRA-2015-gdp_pc.nc"),
        shapefile=Path("data/geometry/BR_Municipios_2023.shp"),
        shapefile_id="CD_MUN",
        output=Path("BRA-2015-gdp_pc.zs.nc"),
        operation="weighted_mean(coverage_weight=area_spherical_km2,default_weight=0)",
        weights=Path("data/worldpop/bra_pop_2015_CN_1km_R2025A_UA_v1.tif"),
        resample="sremapbil",
    )


def test_read_zonalstats_config_multiple():
    config = zonalstats_config1 + "\n" + zonalstats_config2
    assert read_zonalstats_config(config) == [
        ZonalStatsConfig(
            raster=Path("data/country/BRA-2015-gdp_pc.nc"),
            shapefile=Path("data/geometry/BR_Municipios_2023.shp"),
            shapefile_id="CD_MUN",
            output=Path("BRA-2015-gdp_pc.zs.nc"),
            operation="weighted_mean(coverage_weight=area_spherical_km2,default_weight=0)",
            weights=Path("data/worldpop/bra_pop_2015_CN_1km_R2025A_UA_v1.tif"),
            resample="sremapbil",
        ),
        ZonalStatsConfig(
            raster=Path("data/country/BRA-2016-gdp_pc.nc"),
            shapefile=Path("data/geometry/BR_Municipios_2023.shp"),
            shapefile_id="CD_MUN",
            output=Path("BRA-2016-gdp_pc.zs.nc"),
            operation="weighted_mean(coverage_weight=area_spherical_km2,default_weight=0)",
            weights=Path("data/worldpop/bra_pop_2016_CN_1km_R2025A_UA_v1.tif"),
            resample="sremapbil",
        ),
    ]


def test_zonalstats_config_str_simple():
    cfg = ZonalStatsConfig(
        raster=Path("data/raster.nc"),
        shapefile=Path("data/shapefile.shp"),
        shapefile_id="ADMIN",
        output=Path("output.zs.nc"),
        operation="mean(coverage_weight=area_spherical_km2)",
        resample="off",
    )
    assert str(cfg) == (
        "data/raster.nc data/shapefile.shp::ADMIN"
        " --operation=mean(coverage_weight=area_spherical_km2)"
        " --output=output.zs.nc"
    )


def test_zonalstats_config_str_with_weights_and_resample():
    cfg = ZonalStatsConfig(
        raster=Path("data/raster.nc"),
        shapefile=Path("data/shapefile.shp"),
        shapefile_id="ADMIN",
        output=Path("output.zs.nc"),
        operation="weighted_mean(coverage_weight=area_spherical_km2,default_weight=0)",
        weights=Path("data/weights.tif"),
        resample="sremapbil",
    )
    assert str(cfg) == (
        "data/raster.nc data/shapefile.shp::ADMIN"
        " --operation=weighted_mean(coverage_weight=area_spherical_km2,default_weight=0)"
        " --weights=data/weights.tif --resample=sremapbil --output=output.zs.nc"
    )


def test_zonalstats_config_cli_roundtrip():
    cfg = ZonalStatsConfig(
        raster=Path("data/raster.nc"),
        shapefile=Path("data/shapefile.shp"),
        shapefile_id="ADMIN",
        output=Path("output.zs.nc"),
        operation="mean(coverage_weight=area_spherical_km2)",
        resample="off",
    )
    assert ZonalStatsConfig.from_cli(str(cfg)) == cfg


def test_zonalstats_config_cli_roundtrip_with_weights():
    cfg = ZonalStatsConfig(
        raster=Path("data/country/BRA-2015-gdp_pc.nc"),
        shapefile=Path("data/geometry/BR_Municipios_2023.shp"),
        shapefile_id="CD_MUN",
        output=Path("BRA-2015-gdp_pc.zs.nc"),
        operation="weighted_mean(coverage_weight=area_spherical_km2,default_weight=0)",
        weights=Path("data/worldpop/bra_pop_2015_CN_1km_R2025A_UA_v1.tif"),
        resample="sremapbil",
    )
    assert ZonalStatsConfig.from_cli(str(cfg)) == cfg


def test_from_str_dispatches_to_cli():
    cli_str = (
        "data/raster.nc data/shapefile.shp::ADMIN"
        " --operation=mean(coverage_weight=area_spherical_km2)"
        " --output=output.zs.nc"
    )
    assert ZonalStatsConfig.from_str(cli_str) == ZonalStatsConfig(
        raster=Path("data/raster.nc"),
        shapefile=Path("data/shapefile.shp"),
        shapefile_id="ADMIN",
        output=Path("output.zs.nc"),
        operation="mean(coverage_weight=area_spherical_km2)",
        resample="off",
    )


def test_from_str_dispatches_to_logfmt():
    assert ZonalStatsConfig.from_str(zonalstats_config1) == ZonalStatsConfig(
        raster=Path("data/country/BRA-2015-gdp_pc.nc"),
        shapefile=Path("data/geometry/BR_Municipios_2023.shp"),
        shapefile_id="CD_MUN",
        output=Path("BRA-2015-gdp_pc.zs.nc"),
        operation="weighted_mean(coverage_weight=area_spherical_km2,default_weight=0)",
        weights=Path("data/worldpop/bra_pop_2015_CN_1km_R2025A_UA_v1.tif"),
        resample="sremapbil",
    )


def test_zonalstats_config_cli_quoted_paths():
    cfg = ZonalStatsConfig(
        raster=Path("data/my raster.nc"),
        shapefile=Path("data/my shapefile.shp"),
        shapefile_id="ADMIN",
        output=Path("my output.zs.nc"),
        operation="mean(coverage_weight=area_spherical_km2)",
        resample="off",
    )
    assert ZonalStatsConfig.from_cli(str(cfg)) == cfg


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
