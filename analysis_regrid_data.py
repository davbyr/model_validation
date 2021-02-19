"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.0 (20-01-2021)

This is a script for regridding generic netCDF data files using the xesmf
and xarray/dask libraries. The result is a new xarray dataset, which contains
the regridded data, and a weights file for future regridding if necessary.
The resulting dataset is initially stored lazily, using dask operations, and
values will need to be computed using regridded.compute(). The recommended
way to use this script however, is to write the regridded data straight to file
by setting write_regridded = True (default).

This script is modular, so bits can be changed as necessary. A routine for 
reading data on the original grid is needed (read_data_for_regridding_input)
and a routine for reading data on the new grid (read_data_for_regridding_output).
By default, these are set up to read NEMO data as input and OSTIA data as
output. Xarray/Dask chunking, lazy loading and parallel procedures can be
used with XESMF.

To work with the XESMF regridding library, the output from these read routines
must be xarray datasets or dataarrays, with geographical variables named as
'lon' (longitude) and 'lat' (latitude). If data is on a rectilinear grid, then
the corresponding xarray dimensions should also be named the same. Other
dimensions and variables can be named arbitrarily. See the XESMF documentation
and default read routines for more information.

If a grid wants to be defined manually, this can also be done. XESMF can only
handly rectilinear and curvilinear grids, and these should be defined differently.
The same applies when it comes to xarray.dataset format and lon/lat naming
conventions. There are two example routines in this script: 
create_rectilinear_grid() and create_curvilinear_grid().
These can be swapped for read_data_for_regridding_output() to change the output
grid.

The script is in three sections: Global Variables, Functions and then the main
script.
"""

import coast
import xarray as xr
import xesmf as xe
import numpy as np
from dask.diagnostics import ProgressBar

'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # GLOBAL VARIABLES
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
# DEFINE OUTPUT FILES

# The new file containing regridded data
fn_regridded = "/Users/dbyrne/Projects/CO9_AMM15/nemo_clim_regridded_to_ostia2.nc"
# File containing regridding weights, if write_weights is True
fn_weights = "/Users/dbyrne/Projects/CO9_AMM15/regrid_weights.nc"

# Which files to write. 
write_weights = False
write_regridded = True

# Regridding method. Any XESMF method: bilinear, conservative, cubic, etc
interp_method = 'bilinear'

'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # FUNCTIONS
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

def read_data_for_regridding_nemo():
    ''' Ouput from this function must be an xarray.Dataset or xarray.DataArray 
    object. Longitude and latitude variables must be named 'lon' and 'lat' 
    respectively. If data is on rectilinear grid, there should also be 
    dimensions named 'lon' and 'lat'. f using, "mask" should be 1s over the 
    ocean and 0 over the land.
    '''
    
    # NEMO FILES
    fn_nemo_data = '/Users/dbyrne/Projects/CO9_AMM15/data/climatology/p0_seasonal_mean.nc'
    fn_nemo_domain = '/Users/dbyrne/Projects/CO9_AMM15/data/nemo/CO7_EXACT_CFG_FILE.nc'
    
    # Use coast to create NEMO object and extract dataset. Only keep variables
    # of interest.
    ds = coast.NEMO(fn_nemo_data, fn_nemo_domain, multiple=True, 
                  chunks={'season':1})
    ds = ds.dataset[['ssh','temperature','salinity']]
    
    # Rename longitude and latitude for XESMF
    ds = ds.rename({'longitude':'lon', 'latitude':'lat'})
    
    # Create a landmask and place into dataset
    domain = xr.open_dataset(fn_nemo_domain, chunks = {})
    ds['mask'] = (['y_dim','x_dim'],domain.top_level[0].values.astype(bool))
    
    return ds
    
def read_data_for_regridding_ostia():
    ''' Ouput from this function must be an xarray.Dataset or xarray.DataArray 
    object. Longitude and latitude variables must be named 'lon' and 'lat' 
    respectively. If data is on rectilinear grid, there should also be 
    dimensions named 'lon' and 'lat'. If using, "mask" should be 1s over the 
    ocean and 0 over the land.
    '''
    
    # OSTIA FILES
    fn_ostia = "/Users/dbyrne/Projects/CO9_AMM15/data/climatology/ostia_seasonal_mean.nc"
    
    # Read OSTIA data using xarray. Variables are already named appropriately.
    # Extract the first time index of mask and redefine to be a landmask,
    # not a land sea-ice mask.
    ds = xr.open_mfdataset(fn_ostia, chunks = {})
    mask_values = ds.mask.isel(season=0).values.astype(int) == 1
    ds['mask'] = (['lat','lon'], mask_values)
    return ds

def create_rectilinear_grid(lon, lat):
    ''' For manually creating a rectilinear output grid. Not necessary if grid
    already available. Lon and lat should be
    1D numpy arrays define the longitude and latitudes. For example:
        
        lat = np.arange(40,60,0.1)
        lon = np.arange(-10,10,0.1)
    '''
    # Write to xarray Dataset
    ds = xr.Dataset({'lat': (['lat'], lat),
                     'lon': (['lon'], lon),
                    })
    return ds

def create_curvilinear_grid():
    ds = xr.Dataset()
    return ds

def regrid_using_xesmf(ds_in, ds_out, interp_method):
    ''' Create XESMF regridding weights and apply to intput dataset '''
    
    print('Regridding using XESMF..')
    
    # Create the xesmf weights object.
    weights = xe.Regridder(ds_in, ds_out, interp_method)

    # Regrid the input data. Will preserve Dask chunking and lazy loading until
    # .compute() is called.
    regridded = weights(ds_in)
    
    return weights, regridded

def write_weights(weights, fn_weights):
    ''' Write regridding weights to file'''
    print("Writing weights file..")
    with ProgressBar():
        weights.to_netcdf(fn_weights)
    
def write_regridded(regridded, fn_regridded):
    ''' Write regridded dataset to file.'''
    print("Writing regridded data..")
    with ProgressBar():
        regridded.to_netcdf(fn_regridded)
      
'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # MAIN SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

# Define input and output datasets here.
ds_in = read_data_for_regridding_nemo()
ds_out = read_data_for_regridding_ostia()

# Calculate weights and regrid the data
weights, regridded = regrid_using_xesmf(ds_in, ds_out, interp_method)

if write_weights:
    write_weights(weights, fn_weights)

if write_regridded:
    write_regridded(regridded, fn_regridded)