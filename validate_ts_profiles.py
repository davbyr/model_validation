
"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.0 (20-01-2021)

"""

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # GLOBAL VARIABLES
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
# Input paths and Filenames
fn_nemo_data = ''
fn_nemo_domain = ''
fn_en4 = '/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/en4/EN.4.2.1.f.profiles.g10.201901.nc'




 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 #SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


import coast
import coast.general_utils as coastgu
import coast.plot_util as coastpu
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path

def main():
    
    nemo = read_monthly_nemo(fn_)
    
    en4 = read_monthly_profile_en4(fn_en4)
    
    # Monthly Analysis

    

def read_profile_en4(fn_en4):
    '''
    '''
    
    en4 = coast.PROFILE()
    en4 = en4.read_EN4(fn_en4, multiple=True)
    
    return en4

def read_monthly_nemo(fn_temperature):
    '''
    '''

    clim_t = xr.open_dataset(fn_temperature)
    
    return clim_t

def write_stats_to_file(stats, dn_output):
    '''
    Writes the stats xarray dataset to a new netcdf file in the output 
    directory. stats is the dataset created by the calculate_statistics()
    routine. dn_output is the specified output directory for the script.
    '''
    fn_save = run_name + "_stats.nc"
    stats.to_netcdf(os.path.join(dn_output, fn_save))
    return

if __name__ == '__main__':
    main()