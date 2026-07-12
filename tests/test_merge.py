import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geoglue.merge import combine_attrs, variable_merge, _group_datasets, merge_datasets


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

    def test_geoglue_config_deduplicated(self):
        result = combine_attrs(
            [{"geoglue_config": "cfg1"}, {"geoglue_config": "cfg1"}], None
        )
        assert result == {"geoglue_config": "cfg1"}

    def test_geoglue_config_unique_joined_with_newline(self):
        result = combine_attrs(
            [{"geoglue_config": "cfg1"}, {"geoglue_config": "cfg2"}], None
        )
        assert result == {"geoglue_config": "cfg1\ncfg2"}

    def test_geoglue_config_bytes_decoded(self):
        result = combine_attrs(
            [{"geoglue_config": b"cfg1"}, {"geoglue_config": "cfg2"}], None
        )
        assert result == {"geoglue_config": "cfg1\ncfg2"}

    def test_geoglue_config_order_preserved(self):
        result = combine_attrs(
            [
                {"geoglue_config": "cfg3"},
                {"geoglue_config": "cfg1"},
                {"geoglue_config": "cfg2"},
            ],
            None,
        )
        assert result == {"geoglue_config": "cfg3\ncfg1\ncfg2"}


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

    def test_merge_combines_geoglue_config(self, tmp_path):
        t = pd.date_range("2020-01-01", periods=3)
        f1 = tmp_path / "temp.nc"
        f2 = tmp_path / "precip.nc"
        make_dataset(["temp"], t, attrs={"geoglue_config": "cfg1"}).to_netcdf(f1)
        make_dataset(["precip"], t, attrs={"geoglue_config": "cfg2"}).to_netcdf(f2)

        result = merge_datasets([f1, f2])
        assert "cfg1" in result.attrs["geoglue_config"]
        assert "cfg2" in result.attrs["geoglue_config"]

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
