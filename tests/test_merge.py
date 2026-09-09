import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geoglue.merge import (
    _group_datasets,
    _migrate_geoglue_config,
    combine_attrs,
    merge_datasets,
    variable_merge,
)


def make_dataset(
    var_names: list[str],
    time: pd.DatetimeIndex,
    attrs: dict | None = None,
) -> xr.Dataset:
    data = np.ones((len(time), 2, 2))
    ds = xr.Dataset(
        {v: (["time", "lat", "lon"], data) for v in var_names},
        coords={"time": time, "lat": [10.0, 20.0], "lon": [30.0, 40.0]},
    )
    if attrs:
        ds.attrs = attrs
    return ds


# ── combine_attrs ─────────────────────────────────────────────────────────────


class TestCombineAttrs:
    def test_empty_list(self):
        assert combine_attrs([], None) == {}

    def test_none_dicts(self):
        assert combine_attrs([None, None], None) == {}

    def test_regular_key_first_value_wins(self):
        assert combine_attrs([{"source": "a"}, {"source": "b"}], None) == {
            "source": "a"
        }

    def test_none_value_skipped(self):
        assert combine_attrs([{"source": None}, {"source": "b"}], None) == {
            "source": "b"
        }

    def test_all_none_values_key_omitted(self):
        assert combine_attrs([{"source": None}], None) == {}

    def test_key_only_in_some_dicts(self):
        result = combine_attrs([{"a": "1"}, {"b": "2"}], None)
        assert result == {"a": "1", "b": "2"}

    def test_history_deduplicated(self):
        result = combine_attrs([{"history": "step1"}, {"history": "step1"}], None)
        assert result == {"history": "step1"}

    def test_history_unique_lines_joined(self):
        result = combine_attrs([{"history": "step1"}, {"history": "step2"}], None)
        assert result == {"history": "step1\nstep2"}

    def test_history_bytes_decoded(self):
        result = combine_attrs([{"history": b"step1"}, {"history": "step2"}], None)
        assert result == {"history": "step1\nstep2"}

    def test_history_order_preserved(self):
        result = combine_attrs(
            [{"history": "step3"}, {"history": "step1"}, {"history": "step2"}],
            None,
        )
        assert result == {"history": "step3\nstep1\nstep2"}

    def test_history_multiline_deduplicated(self):
        result = combine_attrs(
            [{"history": "step1\nstep2"}, {"history": "step2\nstep3"}], None
        )
        assert result == {"history": "step1\nstep2\nstep3"}

    def test_geoglue_config_migrated_logfmt_to_cli(self):
        logfmt = (
            "raster=data/raster.nc "
            "shapefile=data/shapefile.shp "
            "shapefile_id=ADMIN "
            "output=output.zs.nc "
            "operation=mean(coverage_weight=area_spherical_km2) "
            "resample=off "
        )

        result = _migrate_geoglue_config({"geoglue_config": logfmt})
        assert result == {
            "history": "geoglue zonalstats data/raster.nc data/shapefile.shp::ADMIN"
            " --operation=mean(coverage_weight=area_spherical_km2) --output=output.zs.nc"
        }
        assert "geoglue_config" not in result

    def test_geoglue_config_multiline_logfmt_migrated(self):
        def logfmt(raster: str, output: str) -> str:
            return " ".join(
                [
                    f"raster={raster}",
                    "shapefile=data/shapefile.shp",
                    "shapefile_id=ADMIN",
                    f"output={output}",
                    "operation=mean(coverage_weight=area_spherical_km2)",
                    "resample=off",
                ]
            )

        config_val = (
            logfmt("data/r1.nc", "r1.zs.nc") + "\n" + logfmt("data/r2.nc", "r2.zs.nc")
        )
        result = _migrate_geoglue_config({"geoglue_config": config_val})
        assert result == {
            "history": (
                "geoglue zonalstats data/r1.nc data/shapefile.shp::ADMIN"
                " --operation=mean(coverage_weight=area_spherical_km2) --output=r1.zs.nc\n"
                "geoglue zonalstats data/r2.nc data/shapefile.shp::ADMIN"
                " --operation=mean(coverage_weight=area_spherical_km2) --output=r2.zs.nc"
            )
        }

    def test_geoglue_config_migration_preserves_existing_history(self):
        logfmt = (
            "raster=data/raster.nc "
            "shapefile=data/shapefile.shp "
            "shapefile_id=ADMIN "
            "output=output.zs.nc "
            "operation=mean(coverage_weight=area_spherical_km2) "
            "resample=off "
        )

        result = _migrate_geoglue_config(
            {"geoglue_config": logfmt, "history": "prior step"}
        )
        assert result["history"].endswith("\nprior step")
        assert result["history"].startswith("geoglue zonalstats data/raster.nc")
        assert "geoglue_config" not in result

    def test_geoglue_config_unparseable_kept_as_is(self):
        result = _migrate_geoglue_config({"geoglue_config": "not-valid-logfmt"})
        assert result == {"history": "geoglue zonalstats not-valid-logfmt"}

    def test_geoglue_config_bytes_migrated(self):
        logfmt = b" ".join(
            [
                b"raster=data/raster.nc",
                b"shapefile=data/shapefile.shp",
                b"shapefile_id=ADMIN",
                b"output=output.zs.nc",
                b"operation=mean(coverage_weight=area_spherical_km2)",
                b"resample=off",
            ]
        )
        result = _migrate_geoglue_config({"geoglue_config": logfmt})
        assert result == {
            "history": "geoglue zonalstats data/raster.nc data/shapefile.shp::ADMIN"
            " --operation=mean(coverage_weight=area_spherical_km2) --output=output.zs.nc"
        }

    def test_combine_attrs_migrates_geoglue_config(self):
        result = combine_attrs(
            [{"geoglue_config": "cfg1"}, {"geoglue_config": "cfg2"}], None
        )
        assert result == {"history": "geoglue zonalstats cfg1\ngeoglue zonalstats cfg2"}
        assert "geoglue_config" not in result


# ── variable_merge ────────────────────────────────────────────────────────────


class TestVariableMerge:
    def test_single_variable_files_merged(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f1 = tmp_path / "temp.nc"
        f2 = tmp_path / "precip.nc"
        make_dataset(["temp"], t).to_netcdf(f1)
        make_dataset(["precip"], t).to_netcdf(f2)

        result = variable_merge([f1, f2])
        assert "temp" in result.data_vars
        assert "precip" in result.data_vars

    def test_multi_variable_file_preserved(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f = tmp_path / "multi.nc"
        make_dataset(["temp", "precip"], t).to_netcdf(f)

        result = variable_merge([f])
        assert "temp" in result.data_vars
        assert "precip" in result.data_vars

    def test_mixed_single_and_multi_variable(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f1 = tmp_path / "tp.nc"
        f2 = tmp_path / "uv.nc"
        make_dataset(["tp"], t).to_netcdf(f1)
        make_dataset(["u10", "v10"], t).to_netcdf(f2)

        result = variable_merge([f1, f2])
        assert set(result.data_vars) == {"tp", "u10", "v10"}


# ── _group_datasets ───────────────────────────────────────────────────────────


class TestGroupDatasets:
    def test_same_time_range_grouped_together(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f1 = tmp_path / "temp.nc"
        f2 = tmp_path / "precip.nc"
        make_dataset(["temp"], t).to_netcdf(f1)
        make_dataset(["precip"], t).to_netcdf(f2)

        groups = _group_datasets([f1, f2], "time")
        assert len(groups) == 1
        assert set(groups[0]) == {f1, f2}

    def test_different_time_ranges_separate_groups(self, tmp_path):
        t1 = pd.date_range("2020-01-01", periods=3)
        t2 = pd.date_range("2020-01-04", periods=3)
        f1 = tmp_path / "temp_jan1.nc"
        f2 = tmp_path / "temp_jan4.nc"
        make_dataset(["temp"], t1).to_netcdf(f1)
        make_dataset(["temp"], t2).to_netcdf(f2)

        groups = _group_datasets([f1, f2], "time")
        assert len(groups) == 2

    def test_groups_sorted_chronologically(self, tmp_path):
        t1 = pd.date_range("2020-01-01", periods=3)
        t2 = pd.date_range("2020-01-04", periods=3)
        f1 = tmp_path / "temp_jan1.nc"
        f2 = tmp_path / "temp_jan4.nc"
        make_dataset(["temp"], t1).to_netcdf(f1)
        make_dataset(["temp"], t2).to_netcdf(f2)

        # Pass in reverse order — groups should still be sorted by time
        groups = _group_datasets([f2, f1], "time")
        first_ds = xr.open_dataset(groups[0][0])
        second_ds = xr.open_dataset(groups[1][0])
        assert first_ds.time[0].values < second_ds.time[0].values

    def test_single_file_single_group(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f = tmp_path / "temp.nc"
        make_dataset(["temp"], t).to_netcdf(f)

        groups = _group_datasets([f], "time")
        assert len(groups) == 1
        assert groups[0] == [f]

    def test_mismatched_variables_raises(self, tmp_path):
        t1 = pd.date_range("2020-01-01", periods=3)
        t2 = pd.date_range("2020-01-04", periods=3)
        f1 = tmp_path / "temp.nc"
        f2 = tmp_path / "precip.nc"
        make_dataset(["temp"], t1).to_netcdf(f1)
        make_dataset(["precip"], t2).to_netcdf(f2)

        with pytest.raises(ValueError, match="Variable sets"):
            _group_datasets([f1, f2], "time")

    def test_non_contiguous_raises(self, tmp_path):
        # Three groups needed to detect non-uniform gaps
        t1 = pd.date_range("2020-01-01", periods=3)
        t2 = pd.date_range("2020-01-04", periods=3)
        t3 = pd.date_range("2020-01-09", periods=3)  # gap of 2 days after t2
        f1 = tmp_path / "t1.nc"
        f2 = tmp_path / "t2.nc"
        f3 = tmp_path / "t3.nc"
        make_dataset(["temp"], t1).to_netcdf(f1)
        make_dataset(["temp"], t2).to_netcdf(f2)
        make_dataset(["temp"], t3).to_netcdf(f3)

        with pytest.raises(ValueError, match="not contiguous"):
            _group_datasets([f1, f2, f3], "time")

    def test_three_contiguous_groups(self, tmp_path):
        t1 = pd.date_range("2020-01-01", periods=3)
        t2 = pd.date_range("2020-01-04", periods=3)
        t3 = pd.date_range("2020-01-07", periods=3)
        f1 = tmp_path / "t1.nc"
        f2 = tmp_path / "t2.nc"
        f3 = tmp_path / "t3.nc"
        make_dataset(["temp"], t1).to_netcdf(f1)
        make_dataset(["temp"], t2).to_netcdf(f2)
        make_dataset(["temp"], t3).to_netcdf(f3)

        groups = _group_datasets([f1, f2, f3], "time")
        assert len(groups) == 3


# ── merge_datasets ────────────────────────────────────────────────────────────


class TestMergeDatasets:
    def test_multiple_variables_same_time(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f1 = tmp_path / "temp.nc"
        f2 = tmp_path / "precip.nc"
        make_dataset(["temp"], t).to_netcdf(f1)
        make_dataset(["precip"], t).to_netcdf(f2)

        result = merge_datasets([f1, f2])
        assert "temp" in result.data_vars
        assert "precip" in result.data_vars
        assert len(result.time) == 3

    def test_concatenates_along_time(self, tmp_path):
        t1 = pd.date_range("2020-01-01", periods=3)
        t2 = pd.date_range("2020-01-04", periods=3)
        f1 = tmp_path / "temp1.nc"
        f2 = tmp_path / "temp2.nc"
        make_dataset(["temp"], t1).to_netcdf(f1)
        make_dataset(["temp"], t2).to_netcdf(f2)

        result = merge_datasets([f1, f2])
        assert len(result.time) == 6

    def test_time_values_correct_after_concat(self, tmp_path):
        t1 = pd.date_range("2020-01-01", periods=3)
        t2 = pd.date_range("2020-01-04", periods=3)
        f1 = tmp_path / "temp1.nc"
        f2 = tmp_path / "temp2.nc"
        make_dataset(["temp"], t1).to_netcdf(f1)
        make_dataset(["temp"], t2).to_netcdf(f2)

        result = merge_datasets([f1, f2])
        expected_times = pd.date_range("2020-01-01", periods=6)
        np.testing.assert_array_equal(result.time.values, expected_times.values)

    def test_merge_preserves_attrs(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f = tmp_path / "temp.nc"
        make_dataset(["temp"], t, attrs={"source": "test"}).to_netcdf(f)

        result = merge_datasets([f])
        assert result.attrs.get("source") == "test"

    def test_merge_combines_history(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f1 = tmp_path / "temp.nc"
        f2 = tmp_path / "precip.nc"
        make_dataset(["temp"], t, attrs={"history": "step1"}).to_netcdf(f1)
        make_dataset(["precip"], t, attrs={"history": "step2"}).to_netcdf(f2)

        result = merge_datasets([f1, f2])
        assert "step1" in result.attrs["history"]
        assert "step2" in result.attrs["history"]

    def test_merge_migrates_geoglue_config_to_history(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f1 = tmp_path / "temp.nc"
        f2 = tmp_path / "precip.nc"
        make_dataset(["temp"], t, attrs={"geoglue_config": "cfg1"}).to_netcdf(f1)
        make_dataset(["precip"], t, attrs={"geoglue_config": "cfg2"}).to_netcdf(f2)

        result = merge_datasets([f1, f2])
        assert "geoglue_config" not in result.attrs
        assert "geoglue zonalstats cfg1" in result.attrs["history"]
        assert "geoglue zonalstats cfg2" in result.attrs["history"]

    def test_reverse_order_input_same_result(self, tmp_path):
        t1 = pd.date_range("2020-01-01", periods=3)
        t2 = pd.date_range("2020-01-04", periods=3)
        f1 = tmp_path / "temp1.nc"
        f2 = tmp_path / "temp2.nc"
        make_dataset(["temp"], t1).to_netcdf(f1)
        make_dataset(["temp"], t2).to_netcdf(f2)

        result_fwd = merge_datasets([f1, f2])
        result_rev = merge_datasets([f2, f1])
        np.testing.assert_array_equal(result_fwd.time.values, result_rev.time.values)
