"""Zonal stats task configuration"""

from __future__ import annotations
import typing
import logging
import tomllib as toml
import geopandas as gpd
from dataclasses import dataclass
from pathlib import Path

from geoglue.types import Bbox
from geoglue.util import logfmt_escape

logger = logging.getLogger(__name__)

# Allowed resample operations (extendable)
ResampleType = typing.Literal["remapbil", "remapdis", "off"]

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
    source: Path | None = None

    @staticmethod
    def from_dict(data: dict) -> GeoglueConfig:
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
            out_region[r] = ShapefileConfig(Path(region_path), region_id)
        return GeoglueConfig(operation, out_region)

    @staticmethod
    def nil() -> GeoglueConfig:
        return GeoglueConfig({}, {}, None)

    @staticmethod
    def read_file(file: str | Path) -> GeoglueConfig:
        with open(file, "rb") as fp:
            data = toml.load(fp)
            cfg = GeoglueConfig.from_dict(data)
            return GeoglueConfig(cfg.operation, cfg.region, Path(file))


@dataclass(frozen=True)
class CropConfig:
    "Instantiated version of CropConfigTemplate"

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
    "Instantiated version of ZonalStatsTemplate"

    # top-level
    raster: Path
    shapefile: Path
    shapefile_id: str
    output: Path
    operation: str
    # weights
    weights: Path | None = None
    resample: ResampleType = "off"

    def check_exists(self):
        for f in ["raster", "shapefile", "weights"]:
            if getattr(self, f) and not getattr(self, f).exists():
                raise FileNotFoundError(f"{f} = {getattr(self, f)} file not found")

    def __str__(self):
        _raster = f"raster={logfmt_escape(self.raster)}"
        _shapefile = f"shapefile={logfmt_escape(self.shapefile)}"
        _output = f"output={logfmt_escape(self.output)}"
        _weights = f"weights={logfmt_escape(self.weights)}"
        return " ".join(
            [
                _raster,
                _shapefile,
                f"shapefile_id={self.shapefile_id}",
                _output,
                f"operation={self.operation}",
                _weights,
                f"resample={self.resample}",
            ]
        )


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
