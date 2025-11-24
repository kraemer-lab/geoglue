"""geoglue command-lineOPER interface"""

import datetime
import tempfile
from pathlib import Path

from cdo import Cdo
import click
import xarray as xr
import warnings


from .types import Bbox
from .util import bbox_from_region, read_raster, write_variables, read_geotiff
from .zonalstats import compute_config
from .validate import print_file_stats
from .config import (
    ResampleType,
    ShapefileConfig,
    ZonalStatsConfig,
    read_config,
)

warnings.filterwarnings(
    "ignore",
    message="Spatial reference system of input features does not exactly match",
    category=RuntimeWarning,
)


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


@cli.command("stats", help="Shows statistics for a zonalstats file (.zs.nc)")
@click.argument(
    "files", nargs=-1, type=click.Path(exists=True, dir_okay=False, readable=True)
)
def stats(files: tuple[str]):
    for file in files:
        print_file_stats(Path(file))


@cli.command(
    "crop",
    help="""Crop raster data to region

    INPUT is input raster file. BOUNDS can be a bbox (minx,miny,maxx,maxy),
    a shapefile (.shp) or a raster (.tif) from which bounds are calculated.
""",
)
@click.argument("raster", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument("bounds")
@click.option(
    "--split/--no-split",
    default=True,
    help="Whether to split by variable (default=True)",
)
@click.option(
    "--cover",
    help="Additional raster that the cropped raster should cover",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output path, derived from input raster if omitted",
)
@click.option(
    "-c", "--config", type=click.Path(exists=True, dir_okay=False, readable=True)
)
def crop(
    raster: str,
    bounds: str,
    split: bool = True,
    cover: str | None = None,
    output: str | None = None,
    config: str | None = None,
) -> None:
    """
    geoglue crop <raster> <bbox> [--float-bounds]
    geoglue crop <config>
    """
    raster_p = Path(raster)
    src_rast = xr.open_dataset(raster)
    src_bbox = Bbox.from_xarray(src_rast)
    cfg = read_config(config)
    if bounds in cfg.region:
        # bounds is actually a region name, read that instead
        region = cfg.region[bounds]
        bbox = bbox_from_region(str(region.file), integer_bounds=True)
    else:
        bbox = bbox_from_region(bounds, integer_bounds=True)

    if cover:
        cover_rast = read_raster(cover)
        cover_bbox = Bbox.from_xarray(cover_rast)
        while not bbox > cover_bbox:
            bbox = bbox.enlarge(by=1)
        # Enlarge one more time to ensure grid cells are present
        # on all the edges. This is important for grids that are 1 degree spaced
        # TODO: enlarge by specific number of grid cells depending on source resalution
        bbox = bbox.enlarge(by=1)
    if not (src_bbox > bbox):
        print(f"ERROR: Source bbox {src_bbox} not larger than target bbox {bbox}")
        raise SystemExit(1)
    output_p = (
        Path(output)
        if output
        else raster_p.parent / (raster_p.stem + f".{bbox.safe_name}.nc")
    )
    src_rast = src_rast.sel(latitude=bbox.lat_slice, longitude=bbox.lon_slice)
    vars = list(src_rast.data_vars.keys())
    if len(vars) == 1:
        # only one variable, no need to split
        da = src_rast[vars[0]]
        da.to_netcdf(output_p)
        print(output_p)
    elif split:
        outputs = write_variables(src_rast, output_p)
        print("\n".join("crop " + str(o) for o in outputs))
    else:
        src_rast.to_netcdf(output_p)
        print(output_p)


@cli.command(
    "zonalstats",
    help="""Compute zonal statistics

    RASTER is the source raster that zonal statistics will be computed on.
    REGION specifies a region specified in geoglue configuration, or
    a <shapefile>::<primary_key>
""",
)
@click.argument(
    "raster",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, readable=True),
)
@click.argument("region")
@click.option(
    "--operation", help="exactextract operation to run [default: (weighted_)mean]"
)
@click.option(
    "--weights",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, readable=True),
    help="Path to a weights raster",
)
@click.option(
    "--resample",
    type=click.Choice(["remapdis", "remapbil", "off"], case_sensitive=True),
    default="off",
    show_default=True,
    help="Resampling strategy to apply when required",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output path, derived from input raster if omitted",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="geoglue configuration",
)
@click.option(
    "--tmp",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
    help="Temporary directory for intermediate files",
)
def zonalstats(
    raster: str,
    region: str,
    operation: str | None = None,
    weights: str | None = None,
    resample: ResampleType = "off",
    output: str | None = None,
    config: str | None = None,
    tmp: str | None = None,
) -> None:
    gcfg = read_config(config)
    tmp_path = tmp or gcfg.tmp_path
    if isinstance(tmp_path, str):
        tmp_path = Path(tmp_path)
    if resample not in ["remapbil", "remapdis", "off"]:
        raise ValueError("Unsupported method {resample=}")
    if "::" in region:
        shp = ShapefileConfig.from_str(region)
    else:  # try to find in geoglue config
        if region in gcfg.region:
            shp = gcfg.region[region]
        else:
            raise KeyError(
                f"{region=} not found in configuration and no specific path::id param passed"
            )

    raster_p = Path(raster)
    weights_p = Path(weights) if weights else None
    output_p = Path(output) if output else raster_p.parent / (raster_p.stem + ".zs.nc")
    if weights:
        op = (
            operation
            or "weighted_mean(coverage_weight=area_spherical_km2,default_weight=0)"
        )
    else:
        op = operation or "mean(coverage_weight=area_spherical_km2)"
    if weights and "weighted" not in op:
        print("WARN: Passed weights but operation is not weighted, prefixing!")
        op = "weighted_" + op
    op = gcfg.operation.get(op, op)
    print("config\t\toperation", op)

    cfg = ZonalStatsConfig(
        raster=raster_p,
        shapefile=shp.file,
        shapefile_id=shp.pk,
        output=output_p,
        operation=op,
        weights=weights_p,
        resample=resample,
        tmp_path=tmp_path,
    )
    try:
        cfg.check_exists()
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)
    start_time = datetime.datetime.now(datetime.timezone.utc)
    print(f"zonalstats\tconf={gcfg.source} begin={start_time.isoformat()}")
    da = compute_config(cfg)
    nna = da.isnull().sum().item()
    da.to_netcdf(cfg.output)
    print(f"zonalstats\tNA={nna}", cfg)
    end_time = datetime.datetime.now(datetime.timezone.utc)
    print(
        f"zonalstats\tconf={gcfg.source} end={end_time.isoformat()} elapsed={(end_time - start_time).seconds}s"
    )


@cli.command("griddes", help="Show CDO grid description (griddes) for a file")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, readable=True))
def griddes(file: Path):
    _cdo = Cdo()

    match Path(file).suffix:
        case ".nc":
            print("\n".join(_cdo.griddes(input=str(file))))
        case ".tif":
            da = read_geotiff(file)
            with tempfile.NamedTemporaryFile(prefix="geoglue-", suffix=".nc") as f:
                da.to_netcdf(f.name)
                print("\n".join(_cdo.griddes(input=str(f.name))))


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
