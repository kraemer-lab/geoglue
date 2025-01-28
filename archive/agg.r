# Original code by Moritz Kraemer

# install.packages('terra')
# install.packages('sf')
# install.packages('dplyr')
# install.packages('exactextractr')
setwd("~/ghq/github.com/kraemer-lab/DART-Aggregation")
rm(list=ls())
here::i_am("agg.r")
library(here)

# --- Load Data ---
aedes_raster <- terra::rast(here("data", "aegypti.tif"))

population_raster <- terra::rast(here("data", "vnm_ppp_2020_UNadj_constrained.tif"))
city_polygon_vect <- sf::st_read(here("data", "vnm_adm_gov_20201027", "vnm_admbnda_adm2_gov_20201027.shp"))

# Convert city polygon to terra vector (SpatVector)
#city_polygon_vect <- vect(city_polygon)
#city_polygon_vect <- city_polygon[1,]  # Select the first polygon (adjust if needed)

# Ensure that CRS of all rasters and vector are the same
aedes_raster <- terra::project(aedes_raster, terra::crs(city_polygon_vect))
population_raster <- terra::project(population_raster, terra::crs(city_polygon_vect))

# Crop without masking
aedes_raster <- terra::crop(aedes_raster, city_polygon_vect)
population_raster <- terra::crop(population_raster, city_polygon_vect)

# --- Mask the Aedes and Population rasters with the V3 polygon ---
aedes_masked_total <- terra::mask(aedes_raster, city_polygon_vect)
population_masked_high <- terra::mask(population_raster, city_polygon_vect)

# # Resample population raster to match Aedes resolution (the coarser resolution dataset) --- slight differences between SUMS
# population_masked_low <- terra::resample(population_masked_high, aedes_masked_total, method = "sum")
# 
# # check that sum of the pixels is not too dissimilar
# high_res_sum <- terra::global(population_masked_high, fun = "sum", na.rm = TRUE)
# resampled_sum <- terra::global(population_masked_low, fun = "sum", na.rm = TRUE)
# pou
# high_res_sum
# resampled_sum
# 
# #extract the area of each cell that is contained within each polygon
# x <- exactextractr::exact_extract(aedes_masked_total, city_polygon_vect, coverage_area = FALSE,
#                    weights = population_masked_low, 'weighted_mean')
# # no overlap at all in these cases? (4)
# sum(is.na(x))
