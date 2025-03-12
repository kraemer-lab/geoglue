"""Dataset module"""

import datetime
from pathlib import Path

import pandas as pd

class Dataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.metrics = data.metric.unique()
        self.metric = self.metrics[0] if len(self.metrics) == 1 else None
        if "daily_" in self.metrics[0]:
            self.temporal_scope = "daily"
            self.time = self.data.date.unique()
            years = self.data.date.dt.year
        elif "weekly_" in self.metrics[0]:
            self.temporal_scope = "weekly"
            self.time = self.data.isoweek.unique()
            years = self.data.isoweek.str[:4].map(int)
        else:
            raise ValueError("No temporal scope detected in dataset")
        min_year, max_year = int(years.min()), int(years.max())
        fmt_year = str(min_year) if min_year == max_year else f"{min_year}_{max_year}"
        self.weighted = "unweighted" not in self.metrics[0]
        self.admin_level = int(
            max(
                c.removeprefix("GID_")
                for c in self.data.columns
                if c.startswith("GID_")
            )
        )
        self.iso3 = self.data.ISO3.unique()[0]
        self.geometry = Country(self.iso3).admin(self.admin_level)
        self.filename = (
            f"{self.iso3.upper()}-{self.admin_level}-{fmt_year}-{self.metric}.parquet"
        )

    def __repr__(self):
        return repr(self.data)

    def select(self, at: str):
        if self.temporal_scope == "weekly":
            return self.data[self.data.isoweek == at]
        else:
            at_date = datetime.date.fromisoformat(at)
            return self.data[self.data.date.dt.date == at_date]

    def select_values(self, at: str):
        data = self.select(at)
        return data.value.reset_index(drop=True)

    def plot(self, at: str):
        df = gpd.GeoDataFrame(self.select(at).merge(self.geometry))
        ax = df.plot("value", legend=True)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(df.metric.unique()[0])
        plt.show()

    def weekly(self) -> Dataset:
        assert self.metric is not None
        self.data["isoweek"] = self.data.date.dt.strftime("%G-W%V")
        metric = self.metric.replace("daily_", "weekly_")
        agg = metric.split(".")[-1].removeprefix("weekly_")
        weekgroups = self.data.groupby(
            Country(self.iso3).admin_cols(self.admin_level) + ["isoweek"]
        )
        match agg:
            case "mean":
                df = weekgroups.value.mean()
            case "max":
                df = weekgroups.value.max()
            case "min":
                df = weekgroups.value.min()
            case "sum":
                df = weekgroups.value.sum()
        df = df.reset_index().sort_values("isoweek")  # type: ignore
        df["metric"] = metric
        df["ISO3"] = self.iso3
        return Dataset(df)

    def to_parquet(self, folder: Path = Path(".")):
        return self.data.to_parquet(folder / self.filename, index=False)
