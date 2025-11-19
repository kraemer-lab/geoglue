"""geoglue command-line interface"""

import datetime
from pathlib import Path
from typing import Sequence

import click
import xarray as xr

from .types import Bbox
from .util import write_variables
from .zonalstats import compute_config
from .config import CropConfigTemplate, ZonalStatsTemplate


def parse_params(params: Sequence[str]) -> dict[str, str]:
    kwargs = {}
    for p in params:
        if "=" not in p:
            raise click.ClickException(
                f"Invalid argument '{p}', expected key=value format."
            )
        key, value = p.split("=", 1)
        kwargs[key] = value
    return kwargs


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
@click.argument("config", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument("params", nargs=-1)
def crop(config: str, params: tuple[str]) -> None:
    """
    geoglue crop <config>
    """
    kwargs = parse_params(params)
    tmpl = CropConfigTemplate.read_file(config)
    cfg = tmpl.fill(**kwargs)
    print("crop", cfg)
    src_rast = xr.open_dataset(cfg.raster)
    src_bbox = Bbox.from_xarray(src_rast)
    if not (src_bbox > cfg.bbox):
        print(f"ERROR: Source bbox {src_bbox} not larger than target bbox {cfg.bbox}")
        raise SystemExit(1)
    src_rast = src_rast.sel(latitude=cfg.bbox.lat_slice, longitude=cfg.bbox.lon_slice)
    if not cfg.split:
        src_rast.to_netcdf(cfg.output)
    else:
        outputs = write_variables(src_rast, Path(cfg.output))
        print("\n".join("crop " + str(o) for o in outputs))


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
    kwargs = parse_params(params)
    tmpl = ZonalStatsTemplate.read_file(config)
    cfg = tmpl.fill(**kwargs)
    try:
        cfg.check_exists()
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)
    start_time = datetime.datetime.now(datetime.timezone.utc)
    print(f"zonalstats conf={config} begin={start_time.isoformat()}")
    da = compute_config(cfg)
    nna = da.isnull().sum().item()
    da.to_netcdf(cfg.output)
    print(f"zonalstats NA={nna}", cfg)
    end_time = datetime.datetime.now(datetime.timezone.utc)
    print(
        f"zonalstats conf={config} end={end_time.isoformat()} elapsed={(end_time - start_time).seconds}s"
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
