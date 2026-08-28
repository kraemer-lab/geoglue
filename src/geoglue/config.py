"""Zonal stats task configuration"""
# pyright: reportUnusedCallResult=none, reportAny=none

from __future__ import annotations
import os
import shlex
import typing
import logging
import argparse
import tomllib as toml
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from typing_extensions import override

from geoglue.types import Bbox
from geoglue.util import logfmt_escape

logger = logging.getLogger(__name__)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> typing.NoReturn:
        raise argparse.ArgumentError(None, message)


# Allowed resample operations (extendable)
ResampleType = typing.Literal["remapbil", "remapdis", "sremapbil", "off"]

DEFAULT_PATH = Path("geoglue-config.toml")


@dataclass
class VariableSpec:
    min: float | None = 0.0
    max: float | None = None
    max_na_frac: float = 0.0

    def validate(self) -> None:
        if self.min is not None and self.max is not None:
            if float(self.min) > float(self.max):
                raise ValueError(f"min ({self.min}) > max ({self.max})")
        if not (0.0 <= self.max_na_frac <= 1.0):
            raise ValueError(
                f"max_na_frac must be between 0 and 1 (got {self.max_na_frac})."
            )


@dataclass(frozen=True)
class ShapefileConfig:
    file: Path
    pk: str

    @staticmethod
    def from_str(s: str) -> ShapefileConfig:
        parts = s.split("::")
        if len(parts) != 2:
            raise ValueError(
                "ShapefileConfig.from_str() takes a single argument, in the form of <shapefile_path>::<shapefile_id>"
            )
        return ShapefileConfig(Path(parts[0]), parts[1])


@dataclass(frozen=True)
class GeoglueConfig:
    operation: dict[str, str]
    region: dict[str, ShapefileConfig]
    paths: dict[str, Path]
    source: Path | None = None

    @property
    def tmp_path(self) -> Path | None:
        return self.paths.get("tmp", None)

    @staticmethod
    def from_dict(data: dict) -> GeoglueConfig:
        paths = data.get("paths", {})
        for p in paths:
            paths[p] = Path(os.path.expandvars(Path(paths[p]).expanduser()))
        operation = data.get("operation", {})
        region = data.get("region", {})
        out_region = {}
        for r in region:
            if set(region[r].keys()) != {"file", "pk"}:
                raise KeyError(
                    f"Both 'file' and 'pk' (primary key) must be present for region={r!r}"
                )
            region_path = region[r]["file"]
            region_id = region[r]["pk"]
            # load the data and check if shapefile present
            df = gpd.read_file(region_path)
            if region_id not in df:
                raise ValueError(
                    f"Shapefile {region_path} for region={r!r} does not have ID column {region_id!r}"
                )
            if len(df) != len(df[region_id].unique()):
                raise ValueError(
                    f"Column {region_id!r} is not unique for shapefile {region_path}"
                )
            out_region[r] = ShapefileConfig(Path(region_path), region_id)
        return GeoglueConfig(operation, out_region, paths)

    @staticmethod
    def nil() -> GeoglueConfig:
        return GeoglueConfig({}, {}, {}, None)

    @staticmethod
    def read_file(file: str | Path) -> GeoglueConfig:
        with open(file, "rb") as fp:
            data = toml.load(fp)
            cfg = GeoglueConfig.from_dict(data)
            return GeoglueConfig(cfg.operation, cfg.region, cfg.paths, Path(file))


@dataclass(frozen=True)
class CropConfig:
    "Crop configuration"

    raster: Path
    bbox: Bbox
    output: Path
    split: bool = True

    def check_exists(self):
        if not self.raster.exists():
            raise FileNotFoundError("Raster file {self.raster} not found")

    def __str__(self):
        _raster = logfmt_escape(self.raster)
        _output = logfmt_escape(self.output)
        return f"raster={_raster} bbox={self.bbox} output={_output} split={self.split}"


@dataclass(frozen=True)
class ZonalStatsConfig:
    "Zonal statistics configuration"

    # top-level
    raster: Path
    shapefile: Path
    shapefile_id: str
    output: Path
    operation: str
    # weights
    weights: Path | None = None
    resample: ResampleType = "off"
    tmp_path: Path | None = None
    region: str | None = None
    config: Path | None = None

    def check_exists(self):
        for f in ["raster", "shapefile", "weights"]:
            if getattr(self, f) and not getattr(self, f).exists():
                raise FileNotFoundError(f"{f} = {getattr(self, f)} file not found")

    @override
    def __str__(self):
        if self.region:
            _region = self.region
        else:
            _region = logfmt_escape(self.shapefile) + "::" + self.shapefile_id
        _raster = logfmt_escape(self.raster)
        _output = f"--output={logfmt_escape(self.output)}"
        _weights = f"--weights={logfmt_escape(self.weights)}" if self.weights else ""
        _resample = f"--resample={self.resample}" if self.resample != "off" else ""
        _operation = f"--operation={self.operation}"
        return " ".join(
            x for x in [_raster, _region, _operation, _weights, _resample, _output] if x
        )

    @staticmethod
    def from_str(s: str) -> ZonalStatsConfig:
        if "raster=" in s:
            return ZonalStatsConfig.from_logfmt(s)
        return ZonalStatsConfig.from_cli(s)

    @staticmethod
    def from_cli(s: str) -> ZonalStatsConfig:
        parser = _ArgumentParser()
        parser.add_argument("raster")
        parser.add_argument("region")
        parser.add_argument("--weights")
        parser.add_argument("--output", required=True)
        parser.add_argument("--resample")
        parser.add_argument("--operation")
        parser.add_argument("--config")
        args = parser.parse_args(shlex.split(s))
        _weights = Path(args.weights) if args.weights else None
        _output = Path(args.output)
        _resample = args.resample or "off"
        if args.operation is None:
            _operation = (
                "mean(coverage_weight=area_spherical_km2)"
                if _weights is None
                else "weighted_mean(coverage_weight=area_spherical_km2,default_weight=0)"
            )
        else:
            _operation = args.operation
        config = read_config(args.config)
        if "::" in args.region:
            shp = ShapefileConfig.from_str(args.region)
            _region = None
        else:
            _region = args.region
            try:
                shp = config.region[args.region]
            except KeyError:
                raise KeyError(
                    f"Region {args.region!r} not found in configuration {args.config or DEFAULT_PATH!r}"
                )
        return ZonalStatsConfig(
            raster=Path(args.raster),
            shapefile=shp.file,
            shapefile_id=shp.pk,
            output=_output,
            operation=config.operation.get(_operation, _operation),
            resample=_resample,
            weights=_weights,
            config=config.source,
        )

    @staticmethod
    def from_logfmt(s: str) -> ZonalStatsConfig:
        parts = shlex.split(s)
        kv = {}
        required_keys = [
            "raster",
            "shapefile",
            "shapefile_id",
            "output",
            "operation",
            "resample",
        ]

        for p in parts:
            k, _, v = p.partition("=")
            kv[k] = v
        if any(k not in kv for k in required_keys):
            raise KeyError(f"Missing required key, should have {required_keys}")
        _raster = Path(kv["raster"])
        _shapefile = Path(kv["shapefile"])
        _shapefile_id = kv["shapefile_id"]
        _output = Path(kv["output"])
        _op = kv["operation"]
        _resample = kv["resample"]
        _weights = kv.get("weights")
        _weights = Path(_weights) if isinstance(_weights, str) else None
        return ZonalStatsConfig(
            raster=_raster,
            shapefile=_shapefile,
            shapefile_id=_shapefile_id,
            output=_output,
            operation=_op,
            resample=_resample,
            weights=_weights,
        )


def read_zonalstats_config(
    config: str,
) -> ZonalStatsConfig | list[ZonalStatsConfig] | None:
    lines = config.splitlines()
    if len(lines) == 1:
        return ZonalStatsConfig.from_str(lines[0])
    else:
        return [ZonalStatsConfig.from_str(line) for line in lines]


def read_config(config: str | Path | None) -> GeoglueConfig:
    if isinstance(config, (str, Path)):
        if not Path(config).exists():
            raise FileNotFoundError(
                f"geoglue configuration could not be read from {config!r}"
            )
        return GeoglueConfig.read_file(config)
    elif DEFAULT_PATH.exists():
        return GeoglueConfig.read_file(DEFAULT_PATH)
    else:
        return GeoglueConfig.nil()
