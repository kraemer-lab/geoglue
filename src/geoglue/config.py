"""Zonal stats task configuration"""

from __future__ import annotations
import os
import string
import typing
import logging
import tomllib as toml
from dataclasses import dataclass
from pathlib import Path

from geoglue.types import Bbox
from geoglue.util import bbox_from_region, logfmt_escape

logger = logging.getLogger(__name__)

# Allowed resample operations (extendable)
ResampleType = typing.Literal["remapbil", "remapdis", "off"]


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


def _template(data: dict, key: str) -> string.Template | str | None:
    val = data.get(key)
    if val is None:
        return None
    if "$" not in val:
        return val
    return string.Template(val)


def _get_template_vars(s: string.Template | None) -> set[str]:
    if s is None:
        return set()
    return set(s.get_identifiers())


def _require_keys(data: dict, keys: set[str]):
    data_keys = set(data.keys())
    if not (keys <= data_keys):
        raise KeyError(f"Data {data} missing required keys: {keys - data_keys}")


def _apply_template(s: string.Template, mapping: dict) -> str:
    """
    Expand $var like templates from string `s` using mapping
    """
    if "iso3" in mapping:
        mapping["iso3"] = mapping["iso3"].upper()
        mapping["iso3_lower"] = mapping["iso3"].lower()
    s = string.Template(os.path.expanduser(s.template))
    return s.substitute(mapping)


def _check_vars_match(input_vars: set[str], output_vars: set[str]):
    if "iso3_lower" in input_vars:
        input_vars = (input_vars - {"iso3_lower"}) | {"iso3"}
    if "iso3_lower" in output_vars:
        output_vars = (output_vars - {"iso3_lower"}) | {"iso3"}
    if input_vars != output_vars:
        raise ValueError(f"All variables in input {input_vars} must be in output")


@dataclass(frozen=True)
class CropConfig:
    "Instantiated version of CropConfigTemplate"

    raster: Path
    bbox: Bbox
    output: Path
    split: bool = False

    def check_exists(self):
        if not self.raster.exists():
            raise FileNotFoundError("Raster file {self.raster} not found")

    def __str__(self):
        _raster = logfmt_escape(self.raster)
        _output = logfmt_escape(self.output)
        return f"raster={_raster} bbox={self.bbox} output={_output} split={self.split}"


@dataclass
class CropConfigTemplate:
    """Configuration that can be passed to `geoglue crop`"""

    raster: string.Template
    region: string.Template
    output: string.Template
    integer_bounds: bool
    split: bool = True

    @property
    def template_vars(self) -> set[str]:
        return (
            _get_template_vars(self.raster)
            | _get_template_vars(self.region)
            | _get_template_vars(self.output)
        )

    def fill(self, **kwargs) -> CropConfig:
        """
        Fill in a CropConfigTemplate and return a CropConfig
        Also checks that files exist
        """
        mapping = dict(kwargs)

        filled = {
            "raster": Path(_apply_template(self.raster, mapping)),
            "region": _apply_template(self.region, mapping),
            "output": Path(_apply_template(self.output, mapping)),
        }
        bbox = bbox_from_region(filled["region"], self.integer_bounds)

        return CropConfig(filled["raster"], bbox, filled["output"], self.split)

    @classmethod
    def from_dict(cls, data: dict) -> CropConfigTemplate:
        """
        Reads a dictionary into a ZonalStatsTemplate instance
        """
        # optional keys
        split = bool(data.get("split", True))

        # required keys
        _require_keys(data, {"raster", "region", "output", "integer_bounds"})
        raster = string.Template(data["raster"])
        region = string.Template(data["region"])
        output = string.Template(data["output"])
        integer_bounds = bool(data["integer_bounds"])
        input_vars = _get_template_vars(raster) | _get_template_vars(region)
        output_vars = _get_template_vars(output)
        _check_vars_match(input_vars, output_vars)
        return CropConfigTemplate(raster, region, output, integer_bounds, split)

    @classmethod
    def read_file(cls, path: str | Path) -> CropConfigTemplate:
        """
        Reads a TOML config file into a CropConfigTemplate instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as fp:
            return cls.from_dict(toml.load(fp))


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


@dataclass
class ZonalStatsTemplate:
    # top-level
    raster: string.Template
    shapefile: string.Template
    shapefile_id: str
    output: string.Template
    operation: str  # TODO: check only a single operation
    # weights
    weights: string.Template | None = None
    resample: ResampleType = "off"

    @property
    def template_vars(self) -> set[str]:
        return (
            _get_template_vars(self.raster)
            | _get_template_vars(self.shapefile)
            | _get_template_vars(self.output)
            | _get_template_vars(self.weights)
        )

    def fill(self, **kwargs) -> ZonalStatsConfig:
        """
        Fill in a ZonalStatsTemplate and return a ZonalStatsConfig.
        Also checks that files exist
        """
        mapping = dict(kwargs)

        filled = {
            "raster": Path(_apply_template(self.raster, mapping)),
            "shapefile": Path(_apply_template(self.shapefile, mapping)),
            "output": Path(_apply_template(self.output, mapping)),
            "weights": _apply_template(self.weights, mapping) if self.weights else None,
        }
        if filled["weights"]:
            filled["weights"] = Path(filled["weights"])

        if uninstantiated := [f for f in filled if "$" in str(filled[f])]:
            raise KeyError(f"Incomplete instantiation in field {uninstantiated}")

        # Build a new ZonalConfig copy with replaced string fields, keep variables as-is
        return ZonalStatsConfig(
            filled["raster"],
            filled["shapefile"],
            self.shapefile_id,
            filled["output"],
            self.operation,
            filled["weights"],
            self.resample,
        )

    @classmethod
    def from_dict(cls, data: dict) -> ZonalStatsTemplate:
        """
        Reads a dictionary into a ZonalStatsTemplate instance
        """
        # optional keys
        weights = data.get("weights")
        resample = data.get("resample", "off")
        if resample not in typing.get_args(ResampleType):
            raise ValueError(f"Invalid {resample=}, must be one of {ResampleType}")

        # required keys
        _require_keys(
            data, {"raster", "shapefile", "shapefile_id", "output", "operation"}
        )
        raster = string.Template(data["raster"])
        shapefile = string.Template(data["shapefile"])
        weights = string.Template(weights) if weights else None
        output = string.Template(data["output"])
        input_vars = (
            _get_template_vars(raster)
            | _get_template_vars(shapefile)
            | _get_template_vars(weights)
        )
        output_vars = _get_template_vars(output)
        _check_vars_match(input_vars, output_vars)
        if weights and (
            data["operation"] != "area_weighted_sum"
            and not data["operation"].startswith("weighted_")
        ):
            logger.warning(
                f"{data['operation']=} is not weighted, but weights key present, adding weighted prefix"
            )
            data["operation"] = "weighted_" + data["operation"]
        return ZonalStatsTemplate(
            raster,
            shapefile,
            data["shapefile_id"],
            output,
            data["operation"],
            weights,
            resample,
        )

    @classmethod
    def read_file(cls, path: str | Path) -> ZonalStatsTemplate:
        """
        Reads a TOML config file into a ZonalStatsTemplate instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as fp:
            return cls.from_dict(toml.load(fp))
