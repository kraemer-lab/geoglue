"""geoglue command-line interface"""

import datetime
from pathlib import Path

import click
import xarray as xr
import geopandas as gpd

from .types import Bbox
from .util import read_geotiff, write_variables
from .zonalstats import compute_config
from .config import ZonalStatsTemplate


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(prog_name="geoglue")
@click.option(
    "-v", "--verbose", count=True, help="Increase verbosity (repeat for more)."
)
@click.pass_context
def cli(ctx: click.Context, verbose: int) -> None:
    """
    geoglue — geospatial data processing utilities
    """
    # store verbosity in context for downstream commands (if needed)
    ctx.obj = {"verbose": verbose}


@cli.command(
    "crop",
    help="""Crop raster data to region

    INPUT is input raster file. REGION can be a bbox (minx,miny,maxx,maxy),
    a shapefile (.shp) or a raster (.tif)
""",
)
@click.argument("input", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument(
    "region",
    type=str,
)
@click.argument("output", type=click.Path(dir_okay=False, writable=True))
@click.option("--int", "integer_bounds", is_flag=True, help="Crop to integer bounds")
@click.option("--split", is_flag=True, help="Split cropped file by variable")
def crop(
    input: str, region: str, output: str, integer_bounds: bool, split: bool
) -> None:
    """
    geoglue crop <netcdf-file> <bbox-or-shapefile> <output> [--int]
    """
    if "," in region:
        # try processing as a bbox
        bbox = Bbox.from_string(region)
    elif region.endswith(".shp"):  # crop to shapefile
        vec = gpd.read_file(region)
        bbox = Bbox(*vec.total_bounds)
    elif region.endswith(".nc"):
        rast = xr.open_dataset(region)
        bbox = Bbox.from_xarray(rast)
    elif region.endswith(".tif"):
        rast = read_geotiff(region)
        bbox = Bbox.from_xarray(rast)
    else:
        print(
            "ERROR: Unrecognised file type, you can specify bbox directly as minx,miny,maxx,maxy"
        )
        raise SystemExit(1)
    src_rast = xr.open_dataset(input)
    src_bbox = Bbox.from_xarray(src_rast)
    if integer_bounds:
        bbox = bbox.int()
    if not (src_bbox > bbox):
        print(f"ERROR: Source bbox {src_bbox} not larger than target bbox {bbox}")
        raise SystemExit(1)
    src_rast = src_rast.sel(latitude=bbox.lat_slice, longitude=bbox.lon_slice)
    if not split:
        src_rast.to_netcdf(output)
        print(output)
    else:
        outputs = write_variables(src_rast, Path(output))
        print("\n".join(map(str, outputs)))


@cli.command(
    "zonalstats",
    help="""Run zonal statistics from configuration TOML

    CONFIG is a zonal statistics configuration file which can contain
    templated variables like $year, $dataset. These are expanded out
    using PARAMS which are list of key=value pairs like year=2015 dataset=reanalysis_monthly.tp
""",
)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument("params", nargs=-1)
def zonalstats(config: str, params: tuple[str]) -> None:
    """
    geoglue zonalstats <toml-file> [<p1=v1> <p2=v2> ...]
    """
    kwargs = {}
    for p in params:
        if "=" not in p:
            raise click.ClickException(
                f"Invalid argument '{p}', expected key=value format."
            )
        key, value = p.split("=", 1)
        kwargs[key] = value
    tmpl = ZonalStatsTemplate.read_file(config)
    cfg = tmpl.fill(**kwargs)
    print(
        f"conf={config} begin={datetime.datetime.now(datetime.timezone.utc).isoformat()}"
    )
    da = compute_config(cfg)
    nna = da.isnull().sum().item()
    da.to_netcdf(cfg.output)
    print(f"NA={nna}", cfg)
    print(
        f"conf={config} end={datetime.datetime.now(datetime.timezone.utc).isoformat()}"
    )


def main(argv: list[str] | None = None) -> int:
    """Programmatic entrypoint returning an exit code."""
    try:
        cli.main(args=argv, prog_name="geoglue", standalone_mode=False)
        return 0
    except SystemExit as exc:
        # click raises SystemExit on --help or normal exits; propagate code
        if isinstance(exc.code, int):
            return exc.code
        return 0
    except click.ClickException as ce:
        click.echo(f"Error: {ce.format_message()}", err=True)
        return 2
    except KeyboardInterrupt:
        click.echo("Interrupted by user", err=True)
        return 130
    except Exception as exc:
        click.echo(f"Unexpected error: {type(exc).__name__}: {exc}", err=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
