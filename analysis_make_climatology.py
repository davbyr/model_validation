"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.0 (20-01-2021)
"""

import coast
import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # GLOBAL VARIABLES
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
# Input paths and Filenames
fn_nemo_data = '/Users/dbyrne/Projects/CO9_AMM15/data/nemo/20*'
fn_nemo_domain = '/Users/dbyrne/Projects/CO9_AMM15/data/nemo/CO7_EXACT_CFG_FILE.nc'
fn_out = "/Users/dbyrne/Projects/CO9_AMM15/p0_seasonal_mean.nc"
fn_ostia = "/Users/dbyrne/Projects/CO9_AMM15/data/ostia/*.nc"

climatology_frequency = 'month'
time_var_name = 'time'

###############

def main():
   # ostia = read_ostia(fn_ostia)
    
    data = coast.NEMO(fn_nemo_data, fn_nemo_domain, multiple=True, chunks='auto')
    data = data.dataset[['temperature','salinity','ssh']].isel(z_dim=0)
    
    # COAsT climatology
    CLIM = coast.CLIMATOLOGY()
    clim_mean = CLIM.make_climatology(data, 'season', time_dim_name='t_dim',
                                      fn_out=fn_out)
    
    
    

def read_ostia(fn_ostia):
    kelvin_to_celcius = -273.15
    ostia = xr.open_mfdataset(fn_ostia, chunks='auto', concat_dim='time', 
                          parallel=True)
    ostia = ostia.rename({'analysed_sst':'temperature'})
    ostia['temperature'] = ostia.temperature + kelvin_to_celcius
    ostia.attrs = {}
    return ostia

def __main__():
    main()