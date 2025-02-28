from geoglue.types import CdoGriddes


def test_population_griddes_repr(population_1km):
    assert (
        str(population_1km.griddes)
        == """
gridtype  = lonlat
gridsize  = 1565499
xsize     = 879
ysize     = 1781
xname     = longitude
yname     = latitude
ylongname = "latitude"
yunits    = "degrees_north"
xfirst    = 102.14874960275003
xinc      = 0.0083333333
yfirst    = 8.557916831327146
yinc      = 0.0083333333
xlongname = "longitude"
xunits    = "degrees_east"
  """.strip()
    )


def test_approx_equal_success(population_1km):
    assert population_1km.griddes.approx_equal(
        CdoGriddes(
            gridtype="lonlat",
            gridsize=1565499,
            xsize=879,
            ysize=1781,
            xname="longitude",
            yname="latitude",
            ylongname="latitude",
            yunits="degrees_north",
            xfirst=102.14874960275003,
            xinc=0.0083333333,
            yfirst=8.557916831327146,
            yinc=0.0083333333,
            xlongname="longitude",
            xunits="degrees_east",
        )
    )


def test_approx_equal_fail_float_diff(population_1km):
    assert not population_1km.griddes.approx_equal(
        CdoGriddes(
            gridtype="lonlat",
            gridsize=1565499,
            xsize=879,
            ysize=1781,
            xname="longitude",
            yname="latitude",
            ylongname="latitude",
            yunits="degrees_north",
            xfirst=102.15,
            xinc=0.0083333333,
            yfirst=8.557916831327146,
            yinc=0.0083333333,
            xlongname="longitude",
            xunits="degrees_east",
        )
    )


def test_approx_equal_fail_integer_diff(population_1km):
    assert not population_1km.griddes.approx_equal(
        CdoGriddes(
            gridtype="lonlat",
            gridsize=1565498,
            xsize=879,
            ysize=1781,
            xname="longitude",
            yname="latitude",
            ylongname="latitude",
            yunits="degrees_north",
            xfirst=102.15,
            xinc=0.0083333333,
            yfirst=8.557916831327146,
            yinc=0.0083333333,
            xlongname="longitude",
            xunits="degrees_east",
        )
    )


def test_from_file():
    assert CdoGriddes.from_file(
        "data/VNM/era5/VNM-2020-era5.daily_sum.nc"
    ) == CdoGriddes(
        gridtype="lonlat",
        gridsize=2145,
        xsize=33,
        ysize=65,
        xname="longitude",
        xlongname="longitude",
        xunits="degrees_east",
        yname="latitude",
        ylongname="latitude",
        yunits="degrees_north",
        xfirst=102,
        xinc=0.25,
        yfirst=24,
        yinc=-0.25,
    )
