"Common types used in geoglue"

from __future__ import annotations

import copy
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict

from cdo import Cdo

_cdo = Cdo()


CdoResampling = Enum("CdoResampling", ["remapbil", "remapdis"])

@dataclass
class CdoGriddes:
    "Grid specification"

    gridtype: str
    gridsize: int
    xsize: int
    ysize: int
    xname: str
    yname: str
    ylongname: str
    yunits: str
    xfirst: float
    xinc: float
    yfirst: float
    yinc: float
    xlongname: str = "longitude"
    xunits: str = "degrees_east"

    def __str__(self) -> str:
        out = []
        for k, v in asdict(self).items():
            if k.endswith("units") or k.endswith("longname"):
                out.append(f'{k:9s} = "{v}"')
            else:
                out.append(f"{k:9s} = {v}")
        return "\n".join(out)

    @staticmethod
    def from_file(file: str | Path, base: CdoGriddes | None = None, **kwargs) -> CdoGriddes:
        out = {}
        for line in _cdo.griddes(input=str(file)):
            if line.startswith("#"):
                continue
            key, value = line.split("=")
            key = key.strip()
            value = value.strip()
            if key.endswith("size"):
                out[key] = int(value)
                continue
            if key.endswith("first") or key.endswith("inc"):
                out[key] = float(value)
            else:
                out[key] = value.replace('"', "")
        if base:
            new_out = copy.deepcopy(asdict(base))
            # override keys already found with keys in file
            new_out.update(out)
        else:
            new_out = out
        if kwargs:
            new_out.update(kwargs)
        return CdoGriddes(**out)

    def write(self, file: str | Path):
        Path(file).write_text(str(self))
