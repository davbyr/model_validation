"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.0 (20-01-2021)

This is a script for regridding generic netCDF data files using the xesmf
and xarray/dask libraries. The result is a new xarray dataset, which contains

"""

import coast
import xarray as xr
import numpy as np
import xesmf as xe
from dask.diagnostics import ProgressBar

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # GLOBAL VARIABLES
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
# Input paths and Filenames
fn_nemo_data = '/Users/dbyrne/Projects/CO9_AMM15/p0_seasonal_mean.nc'
fn_nemo_domain = '/Users/dbyrne/Projects/CO9_AMM15/data/nemo/CO7_EXACT_CFG_FILE.nc'

fn_ostia_grid = "/Users/dbyrne/Projects/CO9_AMM15/ostia_seasonal_mean.nc"

fn_regridded = "/Users/dbyrne/Projects/CO9_AMM15/nemo_clim_regridded_to_ostia.nc"
fn_weights = "/Users/dbyrne/Projects/CO9_AMM15/regrid_weights.nc"

write_weights = False
write_regridded = True

interp_method = 'bilinear'

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Define input dataset here.
# This is the data that will be regridded TO something else.
# Longitude and Latitude variables must be named (or renamed) to 'lon' and
# 'lat' to work with xesmf regridding. For rectangular locations, the 
# corresponding xarray dimensions should also carry these names.
ds_in = coast.NEMO(fn_nemo_data, fn_nemo_domain, multiple=True, 
                  chunks={'season':1})
ds_in = ds_in.dataset[['ssh','temperature','salinity']]
ds_in = ds_in.rename({'longitude':'lon', 'latitude':'lat'})

# Create a landmask
domain = xr.open_dataset(fn_nemo_domain, chunks = {})
ds_in['mask'] = (['y_dim','x_dim'],~domain.top_level[0].squeeze())


# Define the locations of the output (regridded) dataset.
# For xesmf, these are defined in the 'ds_out' xarray dataset. The same
# naming conventions for lon and lat apply for this dataset too.
ds_out = xr.open_dataset(fn_ostia_grid, chunks = {})
ds_out['mask'] = ds_out.mask.isel(season=0)

# Create the xesmf regridder object. This contains the weights necessary for
# interpolation
regridder = xe.Regridder(ds_in, ds_out, interp_method)

# Regrid the input data. Will preserve Dask chunking and lazy loading until
# .compute() is called.
dr_out = dr_out = regridder(ds_in)

# Write regridding weights to file
if write_weights:
    regridder.to_netcdf(fn_weights)

# Write regridded dataset to file.
if write_regridded:
    with ProgressBar():
        dr_out.to_netcdf(fn_regridded)