
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
fn_en4 = '/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/en4/EN.4.2.1.f.profiles.g10.20*'
fn_output_data = ""
dn_output_figs = ""

surface_depth = 10

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 #SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


def main():
    print(' *Tidal validation starting.* ')
    
    # Read in NEMO data to NEMO object
    model = read_monthly_model_nemo(fn_nemo_data, fn_nemo_domain)
    print(' 1. Monthly model data read from file(s).')
    
    # Read in Profile data to PROFILE object
    obs = read_profile_monthly_en4(fn_en4)
    print(' 2. Observed profiles read from file(s).')
        
    # ----------------------------------------------------
    # 3. Use only observations that are within model domain
    lonmax = np.nanmax(model['longitude'])
    lonmin = np.nanmin(model['longitude'])
    latmax = np.nanmax(model['latitude'])
    latmin = np.nanmin(model['latitude'])
    ind = coast.general_utils.subset_indices_lonlat_box(obs['longitude'], 
                                                        obs['latitude'],
                                                        lonmin, lonmax, 
                                                        latmin, latmax)[0]
    obs = obs.isel(profile=ind) # FOR TESTING
    print(' 3. Observations subsetted to model domain boundaries')
  
    
    # ----------------------------------------------------
    # 4. Get indices of obs times that fall within model time bounds.
    obs_time_ymd = pd.to_datetime(obs.time.values)
    model_time_ymd = pd.to_datetime(model.time.values)
    
    obs_time_ym = [datetime.datetime(tt.year,tt.month,1) for tt in obs_time_ymd]
    obs_time_ym = np.array(obs_time_ym)
    model_time_ym = [datetime.datetime(tt.year,tt.month,1) for tt in model_time_ymd]
    model_time_ym = np.array(model_time_ym)
    
    min_check = obs_time_ym >= np.nanmin(model_time_ym)
    max_check = obs_time_ym <= np.nanmax(model_time_ym)
    keep_time = np.where( np.logical_and(min_check, max_check) )[0]
    
    obs = obs.isel(profile=keep_time)
    obs_time_ym = obs_time_ym[keep_time]
    print(' 4. Observations subsetted to model time period')
    
    # ----------------------------------------------------
    # 4. Get model indices (space and time) corresponding to observations
    ind2D = coastgu.nearest_indices_2D(model['longitude'], model['latitude'],
                                       obs['longitude'], obs['latitude'])
    print(' 5. Model indices for obs profiles calculated')
    
    # ----------------------------------------------------
    # 5. Load data in month chunks, then analyse one profile at a time
    print(' 5. Starting loop over profiles for analysis..')
    n_prof = obs.dims['profile']
    n_months = len(model.time)
    # Output arrays - For all profiles
    surf_error_tem = np.zeros(n_prof)*np.nan
    surf_error_sal = np.zeros(n_prof)*np.nan
    me_tem = np.zeros(n_prof)*np.nan
    me_sal = np.zeros(n_prof)*np.nan
    mae_tem = np.zeros(n_prof)*np.nan
    mae_sal = np.zeros(n_prof)*np.nan
    cor_tem = np.zeros(n_prof)*np.nan
    cor_sal = np.zeros(n_prof)*np.nan
    
    profile_count = 0
    # Loop over months
    for month_ii in range(0,n_months):
        
        print('Loading month of data: ' + str(model_time_ym[month_ii]))
        pc_complete = np.round(profile_count/n_prof*100,2)
        print(str(pc_complete)+'% of all profiles analyzed.')
        
        # Extract month from model and observations
        model_month = model.isel(t_dim=month_ii).compute()
        time_ind = np.where(obs_time_ym == model_time_ym[month_ii])[0]
        obs_month = obs.isel(profile=time_ind).compute()
        
        # Loop over profiles in this month
        n_prof_month = len(obs_month.time)
        for p_ii in range(0,n_prof_month):
            
            obs_profile = obs_month.isel(profile=p_ii)
            model_profile = model_month.isel(x_dim=ind2D[0][profile_count], 
                                             y_dim=ind2D[1][profile_count])
            
            # Interpolate model to obs
            obs_profile = obs_profile.rename({'depth':'level'})
            model_profile = model_profile.rename({'depth_0':'z_dim'})
            interp_profile = model_profile.interp(z_dim=obs_profile.level, method='linear')
            
            # Get variables into nice numpy arrays
            obs_tem = obs_profile.temperature.to_masked_array()
            obs_sal = obs_profile.practical_salinity.to_masked_array()
            interp_tem = interp_profile.temperature.to_masked_array()
            interp_sal = interp_profile.salinity.to_masked_array()
            
            if interp_tem.mask.all():
                profile_count += 1
                continue
            
            # ----------------------------------------------------
            # Surface stats
            ind_shallowest = np.ma.flatnotmasked_edges(interp_tem)[0]
            tem_diff = obs_tem[ind_shallowest] - interp_tem[ind_shallowest]
            sal_diff = obs_sal[ind_shallowest] - interp_sal[ind_shallowest] 
            surf_error_tem[profile_count] = tem_diff
            surf_error_sal[profile_count] = sal_diff
            
            # ----------------------------------------------------
            # Bottom stats
            
            # ----------------------------------------------------
            # CRPS
            
            # ----------------------------------------------------
            # Whole profile stats
            
            # Errors at all depths
            error_tem = obs_tem - interp_tem
            error_sal = obs_sal - interp_sal
            
            # Absolute errors at all depths
            abs_error_tem = np.abs(error_tem)
            abs_error_sal = np.abs(error_sal)
            
            # Mean errors across depths
            me_tem[profile_count] = np.ma.mean(error_tem)
            me_sal[profile_count] = np.ma.mean(error_sal)
            
            # Mean absolute errors across depths
            mae_tem[profile_count] = np.ma.mean(abs_error_tem)
            mae_sal[profile_count] = np.ma.mean(abs_error_sal)
            
            # Correlations with depth between model and obs
            cor_tem[profile_count] = np.ma.corrcoef(obs_tem, interp_tem)[0,1]
            cor_sal[profile_count] = np.ma.corrcoef(obs_sal, interp_sal)[0,1]
            
            # ----------------------------------------------------
            # Time averaged errors - Average 
            
            # ----------------------------------------------------
            # Area stats
            
            profile_count += 1
            
            
        # Write stats to file, this is done monthly and appended to a 
            
            
    # Make some plots
    

def read_profile_monthly_en4(fn_en4):
    '''
    '''
    
    en4 = coast.PROFILE()
    en4.read_EN4(fn_en4, multiple=True, chunks={})
    
    return en4.dataset

def read_monthly_model_nemo(fn_nemo_dat, fn_nemo_domain):
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

def add_one_month(t):
    """Return a `datetime.date` or `datetime.datetime` (as given) that is
    one month earlier.
    
    Note that the resultant day of the month might change if the following
    month has fewer days:
    
        >>> add_one_month(datetime.date(2010, 1, 31))
        datetime.date(2010, 2, 28)
    """
    import datetime
    one_day = datetime.timedelta(days=1)
    one_month_later = t + one_day
    while one_month_later.month == t.month:  # advance to start of next month
        one_month_later += one_day
    target_month = one_month_later.month
    while one_month_later.day < t.day:  # advance to appropriate day
        one_month_later += one_day
        if one_month_later.month != target_month:  # gone too far
            one_month_later -= one_day
            break
    return one_month_later

if __name__ == '__main__':
    main()