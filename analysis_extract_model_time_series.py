#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 1.0
Date: 25/02/2021
@author: David Byrne (dbyrne@noc.ac.uk)

This script with uses COAsT to extract the nearest model locations to 
a list of longitude/latitude pairs and saves the resulting set of time series 
to file. The script is modular in nature, with 'read' functions, functions
for defining latitude/longitudes and functions for doing the extraction. By 
default, the script is set up to read NEMO data (using COAsT) in an xarray
dataset and extract locations around the UK (Liverpool and Southampton).

Any functions can be changed, and as long as the correct data format is 
adhered to, the rest of the script should continue to work. Model data
should be read into an xarray.dataset object with COAsT dimension and 
coordinate names (dims = (x_dim, y_dim, z_dim, t_dim), coords = (time,
latitude, longitude, depth)). Longitudes and latitudes to extract should be
provided as 1D numpy arrays.


"""
# Import necessary packages

# Uncomment if using a development version of COAsT (git clone)
# import sys
# sys.path.append('<PATH TO COAsT DIRECTORY HERE>')

import coast
import coast.general_utils as general_utils
import numpy as np
import xarray as xr
import os

'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # MAIN SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

def main():
    
    # SET VARIABLES #######################################
    # NEMO data and domain files if using read_model_nemo()
    fn_nemo_data = '<FULL PATH TO NEMO DATA FILE>'
    fn_nemo_domain = '<FULL PATH TO DOMAIN FILE>'

    # Output file to save timeseries -- any existing files will be deleted.
    fn_timeseries = "<FULL PATH TO DESIRED OUTPUT FILE>"
    #######################################################
    
    # Read or create new longitude/latitudes.
    extract_lon, extract_lat = define_locations_to_extract()
    
    # Read data to extract from
    model = read_model_nemo(fn_nemo_data, fn_nemo_domain)
    
    # Extract model locations nearest to extract_lon and extract_lat
    indexed = extract_nearest_points_using_coast(model, extract_lon, extract_lat)
    
    # Write indexed dataset to file
    write_timeseries_to_file(indexed, fn_timeseries)


'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # FUNCTIONS
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

def define_locations_to_extract():
    ''' Routine for defining the lat/lon locations to extract from model data.
    This should return numpy arrays of longitude and latitude. This can be done
    manually or by reading data from another file and extracting lists of 
    locations. '''
    
    # Liverpool and Southampton
    extract_lon = np.array( [-3.018, -1.3833] )
    extract_lat = np.array( [53.45, 50.9 ] )
    
    return extract_lon, extract_lat

def read_model_nemo(fn_nemo_data, fn_nemo_domain):
    ''' Routine for reading NEMO model data using COAsT.
    This should return numpy arrays of longitude and latitude. This can be done
    manually or by reading data from another file and extracting lists of 
    locations'''
    
    # Read NEMO data into a COAsT object (correct format)
    model = coast.NEMO(fn_nemo_data, fn_nemo_domain, grid_ref = 't-grid', 
                       multiple=True, chunks={'time_counter':1})
    # Extract the xarray dataset and desired variables
    model = model.dataset[['temperature','salinity','ssh']]
    
    # Take only the top depth level
    # model = model.isel(z_dim=0)
    
    # Create a landmask and place into dataset
    # Here I create a landmask from the top_level variable in the domain file.
    # This should be named 'landmask'.
    domain = xr.open_dataset(fn_nemo_domain, chunks = {})
    model['landmask'] = (['y_dim','x_dim'],~domain.top_level[0].values.astype(bool))
    
    # If no mask needed, set to None (uncomment below)
    # model['landmask'] = None
    return model

def extract_nearest_points_using_coast(model, extract_lon, extract_lat):
    '''
    Use COAsT to identify nearest model points and extract them into a new
    xarray dataset, ready for writing to file or using directly.
    '''
    # Use COAsT general_utils.nearest_indices_2D routine to work out the model
    # indices we want to extract
    ind2D = general_utils.nearest_indices_2D(model.longitude, model.latitude,
                                             extract_lon, extract_lat,
                                             mask = model.landmask)
    print('Calculated nearest model indices using BallTree.')

    # Extract indices into new array called 'indexed'
    indexed = model.isel(x_dim = ind2D[0], y_dim = ind2D[1])
    
    # Determine distances from extracted locations and save to dataset.
    # Can be used to check points outside of domain or similar problems.
    indexed_dist = general_utils.calculate_haversine_distance(extract_lon, 
                                                          extract_lat, 
                                                          indexed.longitude.values,
                                                          indexed.latitude.values)
    
    # If there is more than one extract location, 'dim_0' will be a dimension
    # in indexed.
    if 'dim_0' in indexed.dims:
        # Rename the index dimension to 'location'
        indexed = indexed.rename({'dim_0':'location'})
        indexed['dist_from_nearest_neighbour'] = ('location', indexed_dist)
    else:
        indexed['dist_from_nearest_neighbour'] = indexed_dist
        
    return indexed

def write_timeseries_to_file(indexed, fn_timeseries):
    ''' Write extracted data to file '''
    if os.path.exists(fn_timeseries):
        os.remove(fn_timeseries)
    print('Writing to file. For large datasets over multiple files, this may \
          take some time')
    indexed.to_netcdf(fn_timeseries)
    
def __main__():
    main()
