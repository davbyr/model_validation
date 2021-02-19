#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:56:50 2021

@author: dbyrne
"""

import coast
import coast.general_utils as general_utils
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import os

'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # FUNCTIONS
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

def define_locations_to_extract():
    
    extract_lon = np.arange(-10,15,2)
    extract_lat = np.arange(40,65,2)
    
    return extract_lon, extract_lat

def read_model_nemo():
    fn_nemo_data = '/Users/dbyrne/Projects/CO9_AMM15/data/nemo/20*'
    fn_nemo_domain = '/Users/dbyrne/Projects/CO9_AMM15/data/nemo/CO7_EXACT_CFG_FILE.nc'
    model = coast.NEMO(fn_nemo_data, fn_nemo_domain, grid_ref = 't-grid', 
                       multiple=True).dataset
    model = model[['temperature','salinity','ssh']]
    
     # Create a landmask and place into dataset
    domain = xr.open_dataset(fn_nemo_domain, chunks = {})
    model['landmask'] = (['y_dim','x_dim'],~domain.top_level[0].values.astype(bool))
    return model

def extract_nearest_points_using_coast(model, extract_lon, extract_lat, 
                                       check_distance=10):
    # Use COAsT general_utils.nearest_indices_2D routine to work out the model
    # indices we want to extract
    ind2D = general_utils.nearest_indices_2D(model.longitude, model.latitude,
                                             extract_lon, extract_lat,
                                             mask = model.landmask)

    indexed = model.isel(x_dim = ind2D[0], y_dim = ind2D[1])

    # Rename the index dimension to 'location'
    indexed = indexed.rename({'dim_0':'location'})
    
    indexed_dist = general_utils.calculate_haversine_distance(extract_lon, 
                                                          extract_lat, 
                                                          indexed.longitude.values,
                                                          indexed.latitude.values)
    locations_to_keep = np.where(indexed_dist <= check_distance)[0]
    # Keep only those locations that are within the critical distance of the 
    # locations to extract (removes points outside of domain)
    indexed = indexed.isel(location = locations_to_keep)
    return indexed

def write_timeseries_to_file(indexed, fn_timeseries):
    ''' Write extracted data to file '''
    if os.path.exists(fn_timeseries):
        os.remove(fn_timeseries)
    with ProgressBar():
        indexed.to_netcdf(fn_timeseries)

'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # SET VARIABLES
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''  

fn_timeseries = "/Users/dbyrne/Projects/CO9_AMM15/data/test.nc"

check_distance = 10

'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # MAIN SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

# Read data to extract from
model = read_model_nemo()

# Read or create new longitude/latitudes
extract_lon, extract_lat = define_locations_to_extract()

# Extract model locations nearest to extract_lon and extract_lat
indexed = extract_nearest_points_using_coast(model, extract_lon, extract_lat,
                                             check_distance)

# Write to file
write_timeseries_to_file(indexed, fn_timeseries)
