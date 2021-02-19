
"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.0 (20-01-2021)

"""

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
import datetime as datetime
import pandas as pd
from scipy import interpolate

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # GLOBAL VARIABLES
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
# Input paths and Filenames
fn_nemo_data = '/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/2007*'
fn_nemo_domain = '/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/CO7_EXACT_CFG_FILE.nc'
fn_2D = '/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/en4/EN.4.2.1.f.profiles.g10.20*'
fn_output_data = ""
dn_output_figs = ""

model_variable_name = ""
obs_variable_name = ""

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 #SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


def main():
    print(' *Tidal validation starting.* ')
    
    # Read in NEMO data to NEMO object
    model = read_climatology_nemo(fn_en4)
    
    # Read in Profile data to PROFILE object
        
    # ----------------------------------------------------
    # 3. Use only observations that are within model domain

            
            
        # Write stats to file, this is done monthly and appended to a 
            
            
    # Make some plots
    

def read_climatology_nemo(fn_en4):
    '''
    '''
    
    en4 = coast.PROFILE()
    en4.read_EN4(fn_en4, multiple=True, chunks={})
    
    return en4.dataset

def read_climatology_OSPIA(fn_nemo_dat, fn_nemo_domain):
    '''
    '''
    nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain, chunks = {}, multiple=True)
    return nemo.dataset[['temperature','salinity', 'e3t','depth_0']]

def write_stats_to_file(stats, dn_output):
    '''
    Writes the stats xarray dataset to a new netcdf file in the output 
    directory. stats is the dataset created by the calculate_statistics()
    routine. dn_output is the specified output directory for the script.
    '''
    return

if __name__ == '__main__':
    main()