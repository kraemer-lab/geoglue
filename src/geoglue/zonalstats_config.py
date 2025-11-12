"""Zonal stats task configuration"""

from __future__ import annotations
import os
import string
import tomllib as toml
from dataclasses import dataclass, field, replace
from pathlib import Path

# Allowed resample operations (extendable)
_ALLOWED_RESAMPLE = {"remapdis", "remapbil", "remapcon", "remapnn", "remapbic", "remap"}


@dataclass
class VariableSpec:
    name: str
    operation: str
    resample: str
    min: float | None = 0.0
    max: float | None = None
    max_na_frac: float = 0.0

    def validate(self) -> None:
        if self.resample is not None and self.resample not in _ALLOWED_RESAMPLE:
            raise ValueError(
                f"Variable '{self.name}': resample '{self.resample}' is not supported."
            )
        if self.min is not None and self.max is not None:
            if float(self.min) > float(self.max):
                raise ValueError(
                    f"Variable '{self.name}': min ({self.min}) > max ({self.max})."
                )
        if not (0.0 <= self.max_na_frac <= 1.0):
            raise ValueError(
                f"Variable '{self.name}': max_na_frac must be between 0 and 1 (got {self.max_na_frac})."
            )


def _template(data: dict, key: str) -> string.Template | str | None:
    val = data.get(key)
    if val is None:
        return None
    if "$" not in val:
        return val
    return string.Template(val)


def _get_template_vars(s: string.Template | str | None) -> set[str]:
    if s is None or isinstance(s, str):
        return set()
    return set(s.get_identifiers())


def _apply_template(s: string.Template | str | None, mapping: dict) -> str | None:
    """
    Expand $var like templates from string `s` using mapping
    """
    if s is None or isinstance(s, str):
        return None

    # First, expand user (~)
    s = string.Template(os.path.expanduser(s.template))
    return s.substitute(mapping)


@dataclass
class ZonalStatsConfig:
    # top-level
    raster: str | string.Template
    shapefile: str | string.Template
    output: str | string.Template

    # includes (kept for reference)
    include: list[str] = field(default_factory=list)

    # crop options
    crop_raster_integer_bounds: bool = False

    # weights
    weights: str | string.Template | None = None
    crop_weights: bool = False
    crop_weights_integer_bounds: bool = False
    resample_direction: str | None = (
        None  # e.g., "raster_to_weights" or "weights_to_raster"
    )

    # resample default
    resample: str = "remapbil"

    # variables
    variables: dict[str, VariableSpec] = field(default_factory=dict)

    # raw data for reference / debugging
    _raw: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def template_vars(self) -> set[str]:
        return (
            _get_template_vars(self.raster)
            | _get_template_vars(self.shapefile)
            | _get_template_vars(self.output)
            | _get_template_vars(self.weights)
        )

    def validate(self) -> None:
        if not self.raster:
            raise ValueError("top-level 'raster' is required and must be non-empty.")
        if not self.shapefile:
            raise ValueError("top-level 'shapefile' is required and must be non-empty.")
        if not self.output:
            raise ValueError("top-level 'output' is required and must be non-empty.")
        if self.resample is not None and self.resample not in _ALLOWED_RESAMPLE:
            raise ValueError(f"resample '{self.resample}' is not supported.")
        for v in self.variables.values():
            v.validate()

    @staticmethod
    def _merge_dicts(base: dict, override: dict) -> dict:
        """
        Merge two dicts recursively where 'override' wins.
        """
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = ZonalStatsConfig._merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _expand_user(pathlike: str | None) -> str | None:
        """Expand ~ (user home). If None, return None."""
        if pathlike is None:
            return None
        return os.path.expanduser(pathlike)

    def instantiate(self, **kwargs) -> ZonalStatsConfig:
        """
        Return a new ZonalConfig with templates filled in for:
          - raster
          - shapefile
          - weights
          - output
        """
        # mapping: keep kwargs as-is (int allowed)
        mapping = dict(kwargs)

        new = {
            "raster": _apply_template(self.raster, mapping),
            "shapefile": _apply_template(self.shapefile, mapping),
            "output": _apply_template(self.output, mapping),
            "weights": _apply_template(self.weights, mapping),
        }

        if uninstantiated := [f for f in new if "$" in str(new[f])]:
            raise KeyError(f"Incomplete instantiation in field {uninstantiated}")

        # Build a new ZonalConfig copy with replaced string fields, keep variables as-is.
        new_cfg = replace(
            self,
            raster=new["raster"],
            shapefile=new["shapefile"],
            output=new["output"],
            weights=new["weights"],
        )
        return new_cfg

    @classmethod
    def read_file(cls, path: str | Path) -> ZonalStatsConfig:
        """
        Read a TOML config file and return a ZonalConfig instance.
        Supports recursive includes, with include = []. Note that
        template variables are only expanded at the top level

        Features:
        - Supports an optional top-level 'include' key (list of file paths). Included files are loaded first,
          then the main file overrides them.
        - Merges nested tables (e.g. [variables]) recursively.
        - Applies defaults and validates fields.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        def _load_single(p: Path) -> dict:
            with p.open("rb") as fp:
                data = toml.load(fp)
            if not isinstance(data, dict):
                raise ValueError(f"TOML root must be a table (dict) in {p}")
            return data

        main_raw = _load_single(path)

        include_key = None
        if "include" in main_raw:
            include_key = "include"

        merged_raw: dict = {}
        visited: set[Path] = set()

        def _resolve_and_merge(p: Path, stack: list[Path]) -> dict:
            rp = p.resolve()
            if rp in visited:
                return {}
            visited.add(rp)

            raw = _load_single(p)

            include_list = []
            if "include" in raw:
                val = raw.get("include")
                if isinstance(val, list):
                    include_list = val
                elif isinstance(val, str):
                    include_list = [val]
                else:
                    raise ValueError(
                        f"In {p}: 'include' must be a string or list of strings if present."
                    )

            merged_here: dict = {}
            for inc in include_list:
                inc_path = (p.parent / inc).resolve()
                if inc_path in stack:
                    raise ValueError(
                        f"Include cycle detected: {' -> '.join(str(x) for x in stack + [inc_path])}"
                    )
                child = _resolve_and_merge(inc_path, stack + [inc_path])
                merged_here = cls._merge_dicts(merged_here, child)

            merged_here = cls._merge_dicts(merged_here, raw)
            return merged_here

        if include_key:
            include_list = main_raw.get(include_key)
            if isinstance(include_list, str):
                include_list = [include_list]
            if include_list is None:
                include_list = []
            if not isinstance(include_list, list):
                raise ValueError(
                    f"Top-level '{include_key}' must be a list of file paths or a single path string."
                )

            merged_raw = {}
            for inc in include_list:
                inc_path = path.parent / inc
                merged_inc = _resolve_and_merge(inc_path, [inc_path])
                merged_raw = cls._merge_dicts(merged_raw, merged_inc)
            merged_raw = cls._merge_dicts(merged_raw, main_raw)
        else:
            merged_raw = _resolve_and_merge(path, [path])

        raw = merged_raw

        raster = _template(raw, "raster")
        shapefile = _template(raw, "shapefile")
        output = _template(raw, "output")

        if raster is None:
            raise ValueError("Required field 'raster' is missing")
        if shapefile is None:
            raise ValueError("Required field 'shapefile' is missing")
        if output is None:
            raise ValueError("Required field 'output' is missing")

        crop_raster_integer_bounds = bool(raw.get("crop_raster_integer_bounds", False))

        weights = raw.get("weights")

        input_vars = (
            _get_template_vars(raster)
            | _get_template_vars(shapefile)
            | _get_template_vars(weights)
        )
        output_vars = _get_template_vars(output) - {"var"}
        if input_vars != output_vars:
            raise ValueError(
                f"All input template variables are not included in output: {input_vars=}, {output_vars=}"
            )

        crop_weights = bool(raw.get("crop_weights", False))
        crop_weights_integer_bounds = bool(
            raw.get("crop_weights_integer_bounds", False)
        )
        resample_direction = raw.get("resample_direction")

        resample = raw.get("resample", "remapbil")
        if resample not in _ALLOWED_RESAMPLE:
            raise ValueError(
                f"Top-level resample '{resample}' not supported. Allowed: {_ALLOWED_RESAMPLE}"
            )

        vars_section = raw.get("variables", {})
        if vars_section is None:
            vars_section = {}

        if not isinstance(vars_section, dict):
            raise ValueError(
                "The 'variables' section must be a table of variable keys."
            )

        variables: dict[str, VariableSpec] = {}
        for var_name, var_cfg in vars_section.items():
            if var_cfg is None:
                var_cfg = {}

            if not isinstance(var_cfg, dict):
                raise ValueError(f"Variable '{var_name}' must be a table of options")

            default_operation = "weighted_mean" if weights else "mean"
            operation = var_cfg.get("operation", default_operation)
            var_resample = var_cfg.get("resample", resample)
            minval = var_cfg.get("min")
            maxval = var_cfg.get("max")
            max_na_frac = var_cfg.get("max_na_frac", 0.0)

            v = VariableSpec(
                name=var_name,
                operation=operation,
                resample=var_resample,
                min=minval,
                max=maxval,
                max_na_frac=float(max_na_frac) if max_na_frac is not None else 0.0,
            )
            variables[var_name] = v

        cfg = cls(
            raster=raster,
            shapefile=shapefile,
            output=output,
            include=main_raw.get(include_key, []) if include_key else [],
            crop_raster_integer_bounds=crop_raster_integer_bounds,
            weights=weights,
            crop_weights=crop_weights,
            crop_weights_integer_bounds=crop_weights_integer_bounds,
            resample_direction=resample_direction,
            resample=resample,
            variables=variables,
            _raw=raw,
        )

        cfg.validate()
        return cfg
