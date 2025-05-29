# geoglue

geoglue is an open-source project developed by the Kraemer Lab at the
University of Oxford, designed to streamline the process of fetching and
aggregating geospatial data to various administrative levels. The tool is
particularly useful for researchers and practitioners working in fields like
epidemiology, climate science, and public health, where spatially resolved data
is essential. Some of the methods are inspired by methods in the
R [{terra}](https://rspatial.github.io/terra/) package that offers easy
to use tools to work with rasters in-memory.

The core functionality of geoglue evolves around automating the (i) retrieval
of geospatial datasets (boundary data from GADM, climate data from ECMWF) and
aggregating them according to administrative boundaries such as countries
(admin0), states or provinces (admin1), and counties or districts (admin2). The
aggregation of raster data, such as temperature or total precipitation to zonal
boundaries is termed *zonal statistics*. We use the
[exactextract](https://isciences.github.io/exactextract/) package to perform
zonal statistics that allows consideration of raster pixels that have a partial
overlap with a zonal polygon.

geoglue has several modules:

- {mod}`geoglue.cds`: Methods to work with data downloaded from ECMWF's [cdsapi](https://cds.climate.copernicus.eu/how-to-api)
- {mod}`geoglue.region`: Fetch geospatial boundaries data from
  [GADM](https://gadm.org) or [geoBoundaries](https://www.geoboundaries.org)
- {mod}`geoglue.memoryraster`: Read and work with GeoTIFF datasets in memory.
- {mod}`geoglue.resample`: Resample data using the
  [Climate Data Operators](https://code.mpimet.mpg.de/projects/cdo) library.
- {mod}`geoglue.zonal_stats`: Perform zonal statistics on climate data, such
  as those downloaded from cdsapi via the `geoglue.cds` module

