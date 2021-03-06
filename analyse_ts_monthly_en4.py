"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.2 (10-03-2021)

A script for validation of monthly mean model temperature and salinity against
EN4 profile data. The input NEMO data should be monthly mean data, with all
depths, and monthly EN4 data as downloaded from:

https://www.metoffice.gov.uk/hadobs/en4/

This script will read in one month at a time. Filenames for the NEMO and EN4
data are generated using the make_nemo_filename() and make_en4_filename()
routines. It is worth checking these functions to make sure that they adhere
to your filenames.

The script will then:
    
    1. Preprocessing
        a. Cut down the EN4 obs to just those over the model domain.
        b. Cit down the EN4 obs to just the model time window.
        c. Initialise output arrays.
        
    2. Pre-Analysis
        a. Loads and analyses data in monthly loads.
        b. Identifies nearest model grid cells to profile data.
           Uses this to extract an equivalent model profile.
        c. Interpolates both profiles to the observations depths (lienarly)
        d. Removes some profiles: Bad interpolation or nearest model point
           too far from observation.
        e. Calculates in situ density and potential density for each profile.
        
    3. Profile Analysis
        **For each profile:
        a. Errors and absolute errors with depth.
        b. Mean error and Mean Absolute Error across depth.
        c. Errors at the surface and bottom (requires bathymetry)
        d. Mixed Layer Depth estimates
        e. Surface Continuous Ranked Probability Score
        
    4. Regional Analysis
       **The script will calculate averages across user defined regions within 
       their model domain. It will also do this for seasonal means.
        a. Mean errors for different regions, binned by depth
    
    5. Postprocessing
        a. Profile analysis written to file monthly.
        b. Regional analysis written to file at the end of script.
        
Output goes into a specified directory. There will be two files for each month
of data: en4_profile_stats and en4_extracted_profiles. The former contains
statistics and the latter contains the interpolate model profiles. There will
also be a single en4_regional_stats.nc file, containing the regional mean
depth-binned errors.
        
"""
# UNCOMMENT IF USING A DEVELOPMENT VERSION OF COAST
import sys
sys.path.append('/home/users/dbyrne/code/COAsT/')

import coast
import coast.general_utils as coastgu
import coast.plot_util as pu
import numpy as np
import datetime as datetime
import pandas as pd
import gsw
import xarray as xr
import sys
import os
import os.path
import glob
from dateutil.relativedelta import *
import scipy.stats as spst
import xarray.ufuncs as uf
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.signal import savgol_filter
from dask.diagnostics import ProgressBar
 
def write_ds_to_file(ds, fn, **kwargs):
        if os.path.exists(fn):
            os.remove(fn)
        ds.to_netcdf(fn, **kwargs)

def read_monthly_profile_en4(fn_en4):
    '''
    '''
    
    en4 = coast.PROFILE()
    en4.read_EN4(fn_en4, chunks={})
    ds = en4.dataset[['potential_temperature','practical_salinity','depth']]
    ds = ds.rename({'practical_salinity':'salinity', 'potential_temperature':'temperature'})
    return ds

def read_monthly_model_nemo(fn_nemo_dat, fn_nemo_domain):
    '''
    '''
    nemo = coast.NEMO(fn_nemo_dat, fn_nemo_domain, chunks={})
    dom = xr.open_dataset(fn_nemo_domain) 
    nemo.dataset['landmask'] = (('y_dim','x_dim'), dom.top_level[0]==0)
    nemo.dataset['bathymetry'] = (('y_dim', 'x_dim'), dom.hbatt[0] )
    ds = nemo.dataset[['temperature','salinity', 'e3t','depth_0', 'landmask','bathymetry', 'bottom_level']]
    return ds.squeeze().load()

def sort_file_names(file_list):
    file_to_read = []
    for file in file_list:
        if '*' in file:
            wildcard_list = glob.glob(file)
            file_to_read = file_to_read + wildcard_list
        else:
            file_to_read.append(file)
            
    # Reorder files to read 
    file_to_read = np.array(file_to_read)
    dates = [os.path.basename(ff) for ff in file_to_read]
    dates = [ff[0:8] for ff in dates]
    dates = [datetime.datetime(int(dd[0:4]), int(dd[4:6]),1) for dd in dates]
    sort_ind = np.argsort(dates)
    file_to_read = file_to_read[sort_ind]
    return file_to_read

def region_def_whole_domain(n_r, n_c):
    return np.ones((n_r,n_c))

def make_nemo_filename(dn, date, suffix):
    month = str(date.month).zfill(2)
    year = date.year
    yearmonth = str(year) + str(month)
    return os.path.join(dn, yearmonth + suffix + '.nc')

def make_en4_filename(dn, date, prefix):
    month = str(date.month).zfill(2)
    year = date.year
    yearmonth = str(year) + str(month)
    return os.path.join(dn, prefix + yearmonth + '.nc')

def analyse_ts_regional(fn_nemo_domain, fn_extracted, fn_out, ref_depth,
                        regional_masks=[], region_names=[]):
    
    ds_ext = xr.open_dataset(fn_extracted, chunks={'profile':10000})
    
    dom = xr.open_dataset(fn_nemo_domain)
    bath = dom.hbatt.values.squeeze()
    dom.close()
    
    regional_masks = regional_masks.copy()
    region_names = region_names.copy()
    regional_masks.append(np.ones(bath.shape))
    region_names.append('whole_domain')
    n_regions = len(regional_masks)
    n_ref_depth = len(ref_depth)
    n_profiles = ds_ext.dims['profile']
    
    ds = ds_ext[['mod_tem','obs_tem','mod_sal','obs_sal','obs_z']].astype('float32')
    ds.load()
    
    ind2D = coastgu.nearest_indices_2D(dom.glamt.values.squeeze(), dom.gphit.values.squeeze(),
                                  ds_ext.longitude.values, ds_ext.latitude.values)
    bathy_pts = bath[ind2D[1], ind2D[0]]
    is_in_region = [mm[ind2D[1], ind2D[0]] for mm in regional_masks]
    is_in_region = np.array(is_in_region, dtype=bool)
    
    
    ds_interp = xr.Dataset(coords = dict(
                               ref_depth = ('ref_depth', ref_depth),
                               longitude = ('profile', ds.longitude.values),
                               latitude = ('profile', ds.latitude.values),
                               time = ('profile', ds.time.values),
                               region = ('region', region_names)),
                           data_vars = dict(
                               bathy = ('profile', bathy_pts),
                               mod_tem = (['profile','ref_depth'], np.zeros((n_profiles, n_ref_depth), dtype='float32')*np.nan),
                               mod_sal = (['profile','ref_depth'], np.zeros((n_profiles, n_ref_depth), dtype='float32')*np.nan),
                               obs_tem = (['profile','ref_depth'], np.zeros((n_profiles, n_ref_depth), dtype='float32')*np.nan),
                               obs_sal = (['profile','ref_depth'], np.zeros((n_profiles, n_ref_depth), dtype='float32')*np.nan)))
    
    
    for pp in range(0, n_profiles):
        prof = ds.isel(profile=pp).swap_dims({'level':'obs_z'}).dropna(dim='obs_z')
        if prof.dims['obs_z']>1:
            try:
                print(pp)
                prof_interp = prof.interp(obs_z = ref_depth)
                dep_len = prof_interp.dims['obs_z']
                ds_interp['mod_tem'][pp, :dep_len] = prof_interp.mod_tem.values
                ds_interp['mod_sal'][pp, :dep_len] = prof_interp.mod_sal.values
                ds_interp['obs_tem'][pp, :dep_len] = prof_interp.obs_tem.values
                ds_interp['obs_sal'][pp, :dep_len] = prof_interp.obs_sal.values
            except:
                print('{0}^^'.format(pp))
                ds_interp['bathy'][pp] = np.nan
        else:
            print('{0}**'.format(pp))
            ds_interp['bathy'][pp] = np.nan
        
    ds_interp['error_tem'] = (ds_interp.mod_tem - ds_interp.obs_tem).astype('float32')
    ds_interp['error_sal'] = (ds_interp.mod_sal - ds_interp.obs_sal).astype('float32')
    
    ds_reg_prof = xr.Dataset(coords = dict(
                                           region = ('region',region_names),
                                           ref_depth = ('ref_depth', ref_depth),
                                           season = ('season', ['DJF','JJA','MAM','SON','All'])))
    ds_reg_prof['prof_mod_tem'] = (['region','season','ref_depth'], np.zeros((n_regions, 5, n_ref_depth), dtype='float32')*np.nan)
    ds_reg_prof['prof_mod_sal'] = (['region','season','ref_depth'], np.zeros((n_regions, 5, n_ref_depth), dtype='float32')*np.nan)
    ds_reg_prof['prof_obs_tem'] = (['region','season','ref_depth'], np.zeros((n_regions, 5, n_ref_depth), dtype='float32')*np.nan)
    ds_reg_prof['prof_obs_sal'] = (['region','season','ref_depth'], np.zeros((n_regions, 5, n_ref_depth), dtype='float32')*np.nan)
    ds_reg_prof['prof_error_tem'] = (['region','season','ref_depth'], np.zeros((n_regions, 5, n_ref_depth), dtype='float32')*np.nan)
    ds_reg_prof['prof_error_sal'] = (['region','season','ref_depth'], np.zeros((n_regions, 5, n_ref_depth), dtype='float32')*np.nan)
    ds_reg_prof['mean_bathy'] = (['region','season'], np.zeros((n_regions, 5))*np.nan)
    
    season_str_dict = {'DJF':0,'JJA':1,'MAM':2,'SON':3}
    
    for reg in range(0,n_regions):
        reg_ind = np.where( is_in_region[reg].astype(bool) )[0]
        reg_tmp = ds_interp.isel(profile = reg_ind)
        reg_tmp_group = reg_tmp.groupby('time.season')
        reg_tmp_mean = reg_tmp_group.mean(dim='profile', skipna=True).compute()
        season_str = reg_tmp_mean.season.values
        season_ind = [season_str_dict[ss] for ss in season_str]
        
        ds_reg_prof['prof_mod_tem'][reg, season_ind] = reg_tmp_mean.mod_tem
        ds_reg_prof['prof_mod_sal'][reg, season_ind] = reg_tmp_mean.mod_sal
        ds_reg_prof['prof_obs_tem'][reg, season_ind] = reg_tmp_mean.obs_tem
        ds_reg_prof['prof_obs_sal'][reg, season_ind] = reg_tmp_mean.obs_sal
        ds_reg_prof['prof_error_tem'][reg, season_ind] = reg_tmp_mean.error_tem
        ds_reg_prof['prof_error_sal'][reg, season_ind] = reg_tmp_mean.error_sal
        ds_reg_prof['mean_bathy'][reg, season_ind] = reg_tmp_mean.bathy
        
        reg_tmp_mean = reg_tmp.mean(dim='profile', skipna=True).compute()
        
        ds_reg_prof['prof_mod_tem'][reg, 4] = reg_tmp_mean.mod_tem
        ds_reg_prof['prof_mod_sal'][reg, 4] = reg_tmp_mean.mod_sal
        ds_reg_prof['prof_obs_tem'][reg, 4] = reg_tmp_mean.obs_tem
        ds_reg_prof['prof_obs_sal'][reg, 4] = reg_tmp_mean.obs_sal
        ds_reg_prof['prof_error_tem'][reg, 4] = reg_tmp_mean.error_tem
        ds_reg_prof['prof_error_sal'][reg, 4] = reg_tmp_mean.error_sal
        ds_reg_prof['mean_bathy'][reg, 4] = reg_tmp_mean.bathy
    
    ds_interp = ds_interp.drop('bathy')
    ds_interp = xr.merge((ds_interp, ds_reg_prof))
    ds_interp['is_in_region'] = (['region','profile'], is_in_region)
    write_ds_to_file(ds_interp, fn_out)

def analyse_ts_per_file(fn_nemo_data, fn_nemo_domain, fn_en4, fn_out,  
                        run_name = 'Undefined', surface_def=5, bottom_def=10, 
                        dist_crit=5, n_obs_levels=400, 
                        model_frequency='daily', instant_data=False):
    
    # ----------------------------------------------------
    # READ 1) Read NEMO, then extract desired variables
    try:   
        # Read NEMO file
        nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain, chunks={'time_counter':1})
        dom = xr.open_dataset(fn_nemo_domain) 
        nemo.dataset['landmask'] = (('y_dim','x_dim'), nemo.dataset.bottom_level==0)
        nemo.dataset['bathymetry'] = (('y_dim', 'x_dim'), dom.hbatt[0] )
        nemo = nemo.dataset[['temperature','salinity', 'e3t','depth_0', 'landmask','bathymetry', 'bottom_level']]
        nemo = nemo.rename({'temperature':'tem','salinity':'sal'})
        mod_time = nemo.time.values
        
    except:
        print('       !!!Problem with NEMO Read: {0}'.format(fn_nemo_data))
        return
        
    # ----------------------------------------------------
    # READ 2) Read EN4, then extract desired variables
    try:
        # Read relevant EN4 files
        en4 = coast.PROFILE()
        en4.read_EN4(fn_en4, chunks={}, multiple=True)
        en4 = en4.dataset[['potential_temperature','practical_salinity','depth']]
        en4 = en4.rename({'practical_salinity':'sal', 'potential_temperature':'tem'})
    except:        
        print('       !!!Problem with EN4 Read: {0}'.format(fn_nemo_data))
        return
    
    print('Files read', flush=True)
    
    # ----------------------------------------------------
    # PREPROC 1) Use only observations that are within model domain
    lonmax = np.nanmax(nemo['longitude'])
    lonmin = np.nanmin(nemo['longitude'])
    latmax = np.nanmax(nemo['latitude'])
    latmin = np.nanmin(nemo['latitude'])
    ind = coast.general_utils.subset_indices_lonlat_box(en4['longitude'], 
                                                        en4['latitude'],
                                                        lonmin, lonmax, 
                                                        latmin, latmax)[0]
    en4 = en4.isel(profile=ind)
    print('EN4 subsetted to model domain', flush=True)
    
    # ----------------------------------------------------
    # PREPROC 2) Use only observations that are within model time window
    en4_time = en4.time.values
    time_max = pd.to_datetime( max(mod_time) ) + relativedelta(hours=12)
    time_min = pd.to_datetime( min(mod_time) ) - relativedelta(hours=12)
    print(time_min)
    print(time_max)
    ind = np.logical_and( en4_time >= time_min, en4_time <= time_max )
    print(np.sum(ind))
    en4 = en4.isel(profile=ind)
    print(en4)
    en4.load()
    print('EN4 subsetted to model time', flush=True)
    
    # ----------------------------------------------------
    # PREPROC 3) Get model indices (space and time) corresponding to observations
    print(en4.dims)
    ind2D = coastgu.nearest_indices_2D(nemo['longitude'], nemo['latitude'],
                                       en4['longitude'], en4['latitude'], 
                                       mask=nemo.landmask)
    
    print('Spatial Indices Calculated', flush=True)
    en4_time = en4.time.values
    ind_time = [ np.argmin( np.abs( mod_time - en4_time[tt] ) ) for tt in range(en4.dims['profile'])]
    min_time = [ np.min( np.abs( mod_time - en4_time[tt] ) ).astype('timedelta64[h]') for tt in range(en4.dims['profile'])]
    ind_time = xr.DataArray(ind_time)
    print('Time Indices Calculated', flush=True)
    
    mod_profiles = nemo.isel(x_dim=ind2D[0], y_dim=ind2D[1], t_dim=ind_time)
    mod_profiles = mod_profiles.rename({'dim_0':'profile'})
    with ProgressBar():
        mod_profiles.load()
    print('Model indexed and loaded', flush=True)
    
    # Define monthly variable arrays for interpolated data
    n_mod_levels = mod_profiles.dims['z_dim']
    n_prof = en4.dims['profile']
    data = xr.Dataset(coords = dict(
                          longitude=(["profile"], en4.longitude.values),
                          latitude=(["profile"], en4.latitude.values),
                          time=(["profile"], en4.time.values),
                          level=(['level'], np.arange(0,n_obs_levels)),
                          ex_longitude = (["profile"], mod_profiles.longitude.values),
                          ex_latitude = (["profile"], mod_profiles.longitude.values),
                          ex_time = (["profile"], mod_profiles.time.values),
                          ex_level = (["ex_level"], np.arange(0, n_mod_levels))),
                      data_vars = dict(
                          mod_tem = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          obs_tem = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          mod_sal = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          obs_sal = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          mod_rho = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          obs_rho = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          mod_s0 = (['profile', 'level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          obs_s0 = (['profile', 'level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          mask_tem = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          mask_sal = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          mask_rho = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          obs_z = (['profile','level'], np.zeros((n_prof , n_obs_levels))*np.nan),
                          ex_mod_tem = (["profile", "ex_level"], mod_profiles.tem.values),
                          ex_mod_sal = (["profile", "ex_level"], mod_profiles.sal.values),
                          ex_depth = (["profile", "ex_level"], mod_profiles.depth_0.values.T)))
    
    

    bad_flag = np.zeros(n_prof).astype(bool)
    
    for prof in range(0,n_prof):
        print(prof)
        # Select the current profile
        mod_profile = mod_profiles.isel(profile = prof)
        obs_profile = en4.isel(profile = prof)
        
        # If the nearest neighbour interpolation is bad, then skip the 
        # vertical interpolation -> keep profile as nans in monthly array
        if all(np.isnan(mod_profile.tem)):
            bad_flag[prof] = True
            continue
        
        # Check that model point is within threshold distance of obs
        # If not, skip vertical interpolation -> keep profile as nans
        interp_dist = coastgu.calculate_haversine_distance(
                                             obs_profile.longitude, 
                                             obs_profile.latitude, 
                                             mod_profile.longitude, 
                                             mod_profile.latitude)
        if interp_dist > dist_crit:
            bad_flag[prof] = True
            continue
        
        # Use bottom_level to mask dry depths
        bl = mod_profile.bottom_level.squeeze().values
        mod_profile = mod_profile.isel(z_dim=range(0,bl))
        
        #
        # Interpolate model to obs depths using a linear interp
        #
        obs_profile = obs_profile.rename({'z_dim':'depth'})
        obs_profile = obs_profile.set_coords('depth')
        mod_profile = mod_profile.rename({'z_dim':'depth_0'})
        
        # If interpolation fails for some reason, skip to next iteration
        try:
            mod_profile_int = mod_profile.interp(depth_0 = obs_profile.depth.values)
        except:
            bad_flag[prof] = True
            continue
        
        # ----------------------------------------------------
        # Analysis 2) Calculate Density per Profile using GSW
        
        # Calculate Density
        ap_obs = gsw.p_from_z( -obs_profile.depth, obs_profile.latitude )
        ap_mod = gsw.p_from_z( -obs_profile.depth, mod_profile_int.latitude )
        # Absolute Salinity            
        sa_obs = gsw.SA_from_SP( obs_profile.sal, ap_obs, 
                                obs_profile.longitude, 
                                obs_profile.latitude )
        sa_mod = gsw.SA_from_SP( mod_profile_int.sal, ap_mod, 
                                mod_profile_int.longitude, 
                                mod_profile_int.latitude )
        # Conservative Temperature
        ct_obs = gsw.CT_from_pt( sa_obs, obs_profile.tem ) 
        ct_mod = gsw.CT_from_pt( sa_mod, mod_profile_int.tem ) 
        
        # In-situ density
        obs_rho = gsw.rho( sa_obs, ct_obs, ap_obs )
        mod_rho = gsw.rho( sa_mod, ct_mod, ap_mod ) 
        
        # Potential Density
        obs_s0 = gsw.sigma0(sa_obs, ct_obs)
        mod_s0 = gsw.sigma0(sa_mod, ct_mod)
        
        # Assign to main array
        data['mod_tem'][prof] = mod_profile_int.tem.values
        data['obs_tem'][prof] = obs_profile.tem.values
        data['mod_sal'][prof] = mod_profile_int.sal.values
        data['obs_sal'][prof] = obs_profile.sal.values
        data['mod_rho'][prof] = mod_rho
        data['obs_rho'][prof] = obs_rho
        data['mod_s0'][prof] = mod_s0
        data['obs_s0'][prof] = obs_s0
        data['obs_z'][prof] = obs_profile.depth
        
    print('       Interpolated Profiles.', flush=True)
    
    # Define seasons
    month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                         7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
    pd_time = pd.to_datetime(data.time.values)
    pd_month = pd_time.month
    season_save = [month_season_dict[ii] for ii in pd_month]
    
    data['season'] = ('profile', season_save)
    data.attrs['run_name'] = run_name
    data['bad_flag'] = ('profile', bad_flag)
    
    # ----------------------------------------------------
    # Analysis 3) All Anomalies
    # Errors at all depths
    data["error_tem"] = (['profile','level'], data.mod_tem - data.obs_tem)
    data["error_sal"] = (['profile','level'], data.mod_sal - data.obs_sal)
    
    # Absolute errors at all depths
    data["abs_error_tem"] = (['profile','level'], np.abs(data.error_tem))
    data["abs_error_sal"] = (['profile','level'], np.abs(data.error_sal))
    
    # ----------------------------------------------------
    # Analysis 7) Whole profile stats
    # Mean errors across depths
    data['me_tem'] = ('profile', np.nanmean(data.error_tem, axis=1))
    data['me_sal'] = ('profile', np.nanmean(data.error_sal, axis=1))
    
    # Mean absolute errors across depths
    data['mae_tem'] = (['profile'], np.nanmean(data.abs_error_tem, axis=1))
    data['mae_sal'] = (['profile'], np.nanmean(data.abs_error_tem, axis=1))
    
    # ----------------------------------------------------
    # Analysis 4) Surface stats
    # Get indices corresponding to surface depth
    # Get averages over surface depths
    data.attrs['surface_definition'] = surface_def
    surface_ind = data.obs_z.values <= surface_def
    
    mod_sal = np.array(data.mod_sal)
    mod_tem = np.array(data.mod_tem)
    obs_sal = np.array(data.obs_sal)
    obs_tem = np.array(data.obs_tem)

    mod_sal[~surface_ind] = np.nan
    mod_tem[~surface_ind] = np.nan
    obs_sal[~surface_ind] = np.nan
    obs_tem[~surface_ind] = np.nan
        
    # Average over surfacedepths
    mod_surf_tem = np.nanmean(mod_tem, axis=1)
    mod_surf_sal = np.nanmean(mod_sal, axis=1)
    obs_surf_tem = np.nanmean(obs_tem, axis=1)
    obs_surf_sal = np.nanmean(obs_sal, axis=1)
 
    # Assign to output arrays
    surf_error_tem =  mod_surf_tem - obs_surf_tem
    surf_error_sal =  mod_surf_sal - obs_surf_sal
    
    data['surf_error_tem'] = (['profile'], surf_error_tem)
    data['surf_error_sal'] = (['profile'], surf_error_sal)
    
    # ----------------------------------------------------
    # Analysis 5) Bottom stats
    # Estimate ocean depth as sum of e3t
    data['bottom_definition'] = bottom_def
    prof_bathy = mod_profiles.bathymetry
    percent_depth = bottom_def
    # Get indices of bottom depths
    bott_ind = data.obs_z.values >= (prof_bathy - percent_depth).values[:,None]
    
    mod_sal = np.array(data.mod_sal)
    mod_tem = np.array(data.mod_tem)
    obs_sal = np.array(data.obs_sal)
    obs_tem = np.array(data.obs_tem)
    
    mod_sal[~bott_ind] = np.nan
    mod_tem[~bott_ind] = np.nan
    obs_sal[~bott_ind] = np.nan
    obs_tem[~bott_ind] = np.nan
        
    # Average over bottom depths
    mod_bott_sal = np.nanmean(mod_sal, axis=1)
    obs_bott_tem = np.nanmean(obs_tem, axis=1)
    mod_bott_tem = np.nanmean(mod_tem, axis=1)
    obs_bott_sal = np.nanmean(obs_sal, axis=1)
    
    data['bott_error_tem'] = (['profile'], mod_bott_tem - obs_bott_tem)
    data['bott_error_sal'] = (['profile'], mod_bott_sal - obs_bott_sal)
    
    print('       Surface and bottom errors done. ', flush=True)
              
    # ----------------------------------------------------
    # WRITE 1) Write monthly stats to file
    
    # Create temp monthly file and write to it
    write_ds_to_file(data, fn_out, mode='w', unlimited_dims='profile')
    
    print('       File Written: ' + fn_out, flush=True)
    
    return 
    
    
    
    
def par_loop_month(parii, current_month, dn_nemo_data, fn_nemo_domain, dn_en4, dn_out, 
               nemo_file_suffix, en4_file_prefix, n_obs_levels, 
               month_season_dict, surface_def, bottom_def, dist_crit, 
               run_name, regional_masks, region_names, depth_bins):
    
    print('Loading month of data: ' + str(current_month), flush=True)
    
    # Make filenames for current month and read NEMO/EN4
    # If this files (e.g. file does not exist), skip to next month
    print(dn_nemo_data)
    try:
        fn_nemo_month = make_nemo_filename(dn_nemo_data, current_month, nemo_file_suffix)
        fn_en4_month = make_en4_filename(dn_en4, current_month, en4_file_prefix)
        print(fn_nemo_month)
        print(fn_en4_month)
        mod_month = read_monthly_model_nemo(fn_nemo_month, fn_nemo_domain)
        obs_month = read_monthly_profile_en4(fn_en4_month)
    except:        
        print('       !!!Problem with read: Not analyzed ')
        return None
    
    # ----------------------------------------------------
    # Init 2) Use only observations that are within model domain
    lonmax = np.nanmax(mod_month['longitude'])
    lonmin = np.nanmin(mod_month['longitude'])
    latmax = np.nanmax(mod_month['latitude'])
    latmin = np.nanmin(mod_month['latitude'])
    ind = coast.general_utils.subset_indices_lonlat_box(obs_month['longitude'], 
                                                        obs_month['latitude'],
                                                        lonmin, lonmax, 
                                                        latmin, latmax)[0]
    obs_month = obs_month.isel(profile=ind)
    obs_month = obs_month.load()
    
    # ----------------------------------------------------
    # Init 4) Get model indices (space and time) corresponding to observations
    ind2D = coastgu.nearest_indices_2D(mod_month['longitude'], 
                                       mod_month['latitude'],
                                       obs_month['longitude'], 
                                       obs_month['latitude'], 
                                       mask=mod_month.landmask)

    n_prof = len(obs_month.time) # Number of profiles in this month
    mod_profiles = mod_month.isel(x_dim=ind2D[0], y_dim=ind2D[1])
    mod_profiles = mod_profiles.rename({'dim_0':'profile'})
    
    # Define regional bools
    n_r = mod_month.dims['y_dim']
    n_c = mod_month.dims['x_dim']
    regional_masks = regional_masks.copy()
    region_names = region_names.copy()
    regional_masks.append(np.ones((n_r, n_c)))
    region_names.append('whole_domain')
    n_regions = len(region_names)
    is_in_region = [mm[ind2D[1], ind2D[0]] for mm in regional_masks]
    
    # Define monthly variable arrays for interpolated data
    mod_tem = np.zeros((n_prof , n_obs_levels))*np.nan
    obs_tem = np.zeros((n_prof, n_obs_levels))*np.nan
    mod_sal = np.zeros((n_prof, n_obs_levels))*np.nan
    obs_sal = np.zeros((n_prof, n_obs_levels))*np.nan
    mod_rho = np.zeros((n_prof, n_obs_levels))*np.nan
    obs_rho = np.zeros((n_prof, n_obs_levels))*np.nan
    mod_s0 = np.zeros((n_prof, n_obs_levels))*np.nan
    obs_s0 = np.zeros((n_prof, n_obs_levels))*np.nan
    obs_z = np.zeros((n_prof, n_obs_levels))*np.nan
    
    bin_widths = depth_bins[1:] - depth_bins[:-1]
    ref_depth = depth_bins[:-1] + .5*bin_widths
    n_ref_depth_bins = len(depth_bins)
    n_ref_depth = len(ref_depth)
    
    mod_tem_bin = np.zeros((n_prof , n_ref_depth))*np.nan
    mod_sal_bin = np.zeros((n_prof , n_ref_depth))*np.nan
    obs_tem_bin = np.zeros((n_prof , n_ref_depth))*np.nan
    obs_sal_bin = np.zeros((n_prof , n_ref_depth))*np.nan

    # Loop over profiles, interpolate model to obs depths and store in
    # monthly arrays
    ind_prof_use = []
    
    for prof in range(0,n_prof):
        
        # Select the current profile
        mod_profile = mod_profiles.isel(profile = prof)
        obs_profile = obs_month.isel(profile = prof)
        
        # If the nearest neighbour interpolation is bad, then skip the 
        # vertical interpolation -> keep profile as nans in monthly array
        if all(np.isnan(mod_profile.temperature)):
            continue
        
        # Check that model point is within threshold distance of obs
        # If not, skip vertical interpolation -> keep profile as nans
        interp_dist = coastgu.calculate_haversine_distance(
                                             obs_profile.longitude, 
                                             obs_profile.latitude, 
                                             mod_profile.longitude, 
                                             mod_profile.latitude)
        if interp_dist > dist_crit:
            continue
        
        # Use bottom_level to mask dry depths
        bl = mod_profile.bottom_level.squeeze().values
        mod_profile = mod_profile.isel(z_dim=range(0,bl))
        
        #
        # Interpolate model to obs depths using a linear interp
        #
        obs_profile = obs_profile.rename({'z_dim':'depth'})
        obs_profile = obs_profile.set_coords('depth')
        mod_profile = mod_profile.rename({'z_dim':'depth_0'})
        
        # If interpolation fails for some reason, skip to next iteration
        try:
            mod_profile_int = mod_profile.interp(depth_0 = obs_profile.depth.values)
        except:
            continue
        
        # Bin profile data by depth bins
    
        mod_tem_bin[prof] = spst.binned_statistic(obs_profile.depth.values, 
                                            mod_profile_int.temperature.values, 
                                            'mean', depth_bins)[0]
        mod_sal_bin[prof] = spst.binned_statistic(obs_profile.depth.values, 
                                            mod_profile_int.salinity.values, 
                                            'mean', depth_bins)[0]
        obs_tem_bin[prof] = spst.binned_statistic(obs_profile.depth.values, 
                                            obs_profile.temperature.values, 
                                            'mean', depth_bins)[0]
        obs_sal_bin[prof] = spst.binned_statistic(obs_profile.depth.values, 
                                            obs_profile.salinity.values, 
                                            'mean', depth_bins)[0]
        
        # ----------------------------------------------------
        # Analysis 2) Calculate Density per Profile using GSW
        
        # Calculate Density
        ap_obs = gsw.p_from_z( -obs_profile.depth, obs_profile.latitude )
        ap_mod = gsw.p_from_z( -obs_profile.depth, mod_profile_int.latitude )
        # Absolute Salinity            
        sa_obs = gsw.SA_from_SP( obs_profile.salinity, ap_obs, 
                                obs_profile.longitude, 
                                obs_profile.latitude )
        sa_mod = gsw.SA_from_SP( mod_profile_int.salinity, ap_mod, 
                                mod_profile_int.longitude, 
                                mod_profile_int.latitude )
        # Conservative Temperature
        ct_obs = gsw.CT_from_pt( sa_obs, obs_profile.temperature ) 
        ct_mod = gsw.CT_from_pt( sa_mod, mod_profile_int.temperature ) 
        
        # In-situ density
        obs_rho_tmp = gsw.rho( sa_obs, ct_obs, ap_obs )
        mod_rho_tmp = gsw.rho( sa_mod, ct_mod, ap_mod ) 
        
        # Potential Density
        obs_s0_tmp = gsw.sigma0(sa_obs, ct_obs)
        mod_s0_tmp = gsw.sigma0(sa_mod, ct_mod)
        
        # Assign monthly array
        mod_tem[prof] = mod_profile_int.temperature.values
        obs_tem[prof] = obs_profile.temperature.values
        mod_sal[prof] = mod_profile_int.salinity.values
        obs_sal[prof] = obs_profile.salinity.values
        mod_rho[prof] = mod_rho_tmp
        obs_rho[prof] = obs_rho_tmp
        mod_s0[prof] = mod_s0_tmp
        obs_s0[prof] = obs_s0_tmp
        obs_z[prof] = obs_profile.depth
        
        # If got to this point then keep the profile
        ind_prof_use.append(prof)
        
    print('       Interpolated Profiles.', flush=True)
    # Find the union of masks for each variable
    mask_tem = np.logical_or(np.isnan(mod_tem), np.isnan(obs_tem))
    mask_sal = np.logical_or(np.isnan(mod_sal), np.isnan(obs_sal))
    mask_rho = np.logical_or(np.isnan(mod_rho), np.isnan(obs_rho))
    
    mod_tem[mask_tem] = np.nan
    obs_tem[mask_tem] = np.nan
    mod_sal[mask_sal] = np.nan
    obs_sal[mask_sal] = np.nan
    
    # Monthly stats xarray dataset - for file output and easy indexing
    data = xr.Dataset(coords = dict(
                              longitude=(["profile"], obs_month.longitude),
                              latitude=(["profile"], obs_month.latitude),
                              time=(["profile"], obs_month.time),
                              level=(['level'], np.arange(0,400)),
                              region = (['region'], region_names)),
                          data_vars = dict(
                              mod_tem = (['profile','level'], mod_tem),
                              obs_tem = (['profile','level'], obs_tem),
                              mod_sal = (['profile','level'], mod_sal),
                              obs_sal = (['profile','level'], obs_sal),
                              mod_rho = (['profile','level'], mod_rho),
                              obs_rho = (['profile','level'], obs_rho),
                              mod_s0 = (['profile', 'level'], mod_s0),
                              obs_s0 = (['profile', 'level'], obs_s0),
                              mask_tem = (['profile','level'], mask_tem),
                              mask_sal = (['profile','level'], mask_sal),
                              mask_rho = (['profile','level'], mask_rho),
                              obs_z = (['profile','level'], obs_z),
                              is_in_region = (['region','profile'], is_in_region)))
    
    analysis = xr.Dataset(coords = dict(
                              longitude=(["profile"], obs_month.longitude),
                              latitude=(["profile"], obs_month.latitude),
                              time=(["profile"], obs_month.time),
                              level=(['level'], np.arange(0,400)),
                              bin_depth = (['bin_depth'], ref_depth),
                              region = (['region'], region_names)),
                      data_vars = dict(
                              obs_z = (['profile','level'], obs_z),
                              mod_tem_binned = (['profile','bin_depth'], mod_tem_bin),
                              mod_sal_binned = (['profile','bin_depth'], mod_sal_bin),
                              obs_tem_binned = (['profile','bin_depth'], obs_tem_bin),
                              obs_sal_binned = (['profile','bin_depth'], obs_sal_bin),
                              is_in_region = (['region','profile'], is_in_region)))
    
    # Keep the profiles we want to keep
    n_prof_use = len(ind_prof_use)
    analysis = analysis.isel(profile = ind_prof_use)
    data = data.isel(profile = ind_prof_use)
    
    # Define season
    month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                         7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
    current_season = month_season_dict[current_month.month]
    season_save = np.ones(n_prof_use)*current_season
    
    analysis['season'] = ('profile', season_save)
    data['season'] = ('profile', season_save)
    data.attrs['run_name'] = run_name
    analysis.attrs['run_name'] = run_name
    
    # ----------------------------------------------------
    # Analysis 3) All Anomalies
    # Errors at all depths
    analysis["error_tem"] = data.mod_tem - data.obs_tem
    analysis["error_sal"] = data.mod_sal - data.obs_sal
    analysis["error_tem_binned"] = analysis.mod_tem_binned - analysis.obs_tem_binned
    analysis["error_sal_binned"] = analysis.mod_sal_binned - analysis.obs_sal_binned
    
    # Absolute errors at all depths
    analysis["abs_error_tem"] = np.abs(analysis.error_tem)
    analysis["abs_error_sal"] = np.abs(analysis.error_sal)
    
    # ----------------------------------------------------
    # Analysis 7) Whole profile stats
    # Mean errors across depths
    
    analysis['me_tem'] = ('profile', np.nanmean(analysis.error_tem, axis=1))
    analysis['me_sal'] = ('profile', np.nanmean(analysis.error_sal, axis=1))
    
    # Mean absolute errors across depths
    analysis['mae_tem'] = (['profile'], 
                           np.nanmean(analysis.abs_error_tem, axis=1))
    analysis['mae_sal'] = (['profile'], 
                           np.nanmean(analysis.abs_error_tem, axis=1))
    
    print('       Basic errors done. ', flush=True)
    
    # ----------------------------------------------------
    # Analysis 4) Surface stats
    # Get indices corresponding to surface depth
    # Get averages over surface depths
    analysis['surface_definition'] = surface_def
    surface_ind = data.obs_z.values <= surface_def
    
    mod_sal_tmp = np.array(data.mod_sal)
    mod_tem_tmp = np.array(data.mod_tem)
    obs_sal_tmp = np.array(data.obs_sal)
    obs_tem_tmp = np.array(data.obs_tem)

    mod_sal_tmp[~surface_ind] = np.nan
    mod_tem_tmp[~surface_ind] = np.nan
    obs_sal_tmp[~surface_ind] = np.nan
    obs_tem_tmp[~surface_ind] = np.nan
        
    # Average over surfacedepths
    mod_surf_tem = np.nanmean(mod_tem_tmp, axis=1)
    mod_surf_sal = np.nanmean(mod_sal_tmp, axis=1)
    obs_surf_tem = np.nanmean(obs_tem_tmp, axis=1)
    obs_surf_sal = np.nanmean(obs_sal_tmp, axis=1)
 
    # Assign to output arrays
    surf_error_tem =  mod_surf_tem - obs_surf_tem
    surf_error_sal =  mod_surf_sal - obs_surf_sal
    
    analysis['surf_error_tem'] = (['profile'], surf_error_tem)
    analysis['surf_error_sal'] = (['profile'], surf_error_sal)
    
    # ----------------------------------------------------
    # Analysis 5) Bottom stats
    # Estimate ocean depth as sum of e3t
    analysis['bottom_definition'] = bottom_def
    prof_bathy = mod_profiles.bathymetry.isel(profile=ind_prof_use)
    percent_depth = bottom_def
    # Get indices of bottom depths
    bott_ind = data.obs_z.values >= (prof_bathy - percent_depth).values[:,None]
    
    mod_sal_tmp = np.array(data.mod_sal)
    mod_tem_tmp = np.array(data.mod_tem)
    obs_sal_tmp = np.array(data.obs_sal)
    obs_tem_tmp = np.array(data.obs_tem)
    
    mod_sal_tmp[~bott_ind] = np.nan
    mod_tem_tmp[~bott_ind] = np.nan
    obs_sal_tmp[~bott_ind] = np.nan
    obs_tem_tmp[~bott_ind] = np.nan
        
    # Average over bottom depths
    mod_bott_sal = np.nanmean(mod_sal_tmp, axis=1)
    obs_bott_tem = np.nanmean(obs_tem_tmp, axis=1)
    mod_bott_tem = np.nanmean(mod_tem_tmp, axis=1)
    obs_bott_sal = np.nanmean(obs_sal_tmp, axis=1)
    
    analysis['bott_error_tem'] = (['profile'], mod_bott_tem - obs_bott_tem)
    analysis['bott_error_sal'] = (['profile'], mod_bott_sal - obs_bott_sal)
    
    print('       Surface and bottom errors done. ', flush=True)
              
    # ----------------------------------------------------
    # WRITE 1) Write monthly stats to file
    
    # Postproc 1) Write data to file for each month - profiles stats
    yy = str(current_month.year)
    mm = str(current_month.month)
    
    # Create temp monthly file and write to it
    fn_tmp = 'en4_stats_by_profile_{0}{1}_{2}.nc'.format(yy,mm.zfill(2),run_name)
    fn_sta = os.path.join(dn_out, fn_tmp)
    write_ds_to_file(analysis, fn_sta, mode='w', unlimited_dims='profile')
    
    print('       File Written: ' + fn_tmp, flush=True)
        
        
    # ----------------------------------------------------
    # WRITE 2) Write monthly extracted profiles to file
    
    # Create temp monthly file and write to it
    fn_tmp = 'en4_extracted_profiles_{0}{1}_{2}.nc'.format(yy,mm.zfill(2), run_name)
    fn_ext = os.path.join(dn_out, fn_tmp)
    write_ds_to_file(data, fn_ext, mode='w', unlimited_dims='profile')
    
    print('       File Written: ' + fn_tmp, flush=True)
    
    return (parii, fn_sta, fn_ext)

def par_loop_regional(rr, ss, fn_stats):
    
    stats = xr.open_dataset(fn_stats, chunks={'profile':10000})
    
    month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                         7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
    pd_time = pd.to_datetime(stats.time.values)
    pd_month = pd_time.month
    pd_season = np.array([month_season_dict[mm] for mm in pd_month])
    
    stats = stats.drop_vars(['obs_z','error_tem','error_sal','abs_error_tem',
                             'abs_error_sal'])
    
    ss_ind = (pd_season==ss).squeeze()
    rr_ind = (stats.is_in_region.isel(region=rr)).squeeze().astype(bool)
            
    if all(~ss_ind) or all(~rr_ind):
        return (None, None, None)
            
    ind = np.where( np.logical_and(ss_ind, rr_ind) )[0]
    stats_tmp = stats.isel(profile=ind)
    stats_mean = stats_tmp.mean(dim='profile', skipna=True).compute()
    stats.close()

    return (rr, ss, stats_mean)

def analyse_ts_monthly_en4(dn_nemo_data, fn_nemo_domain, dn_en4, dn_out, 
                 start_month, end_month, run_name, n_proc=1,
                 nemo_file_suffix = '01_monthly_grid_T',
                 en4_file_prefix = 'EN.4.2.1.f.profiles.g10.',
                 regional_masks=[], region_names = [],
                 surface_def = 5, bottom_def = 10, mld_ref_depth = 5,
                 mld_threshold = 0.02, dist_crit=5, 
                 depth_bins = np.arange(-0.001,200,5)):

    # Define a counter which keeps track of the current month
    n_months = (end_month.year - start_month.year)*12 + \
                    (end_month.month - start_month.month) + 1
    month_list = [start_month + relativedelta(months=+mm) for mm in range(0,n_months)]
    
    # Define a log file to output progress
    print(' *Profile analysis starting.*')
    
    # Number of levels in EN4 data
    n_obs_levels = 400
    
    # Define seasons for seasonal averaging/collation
    month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                         7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
    
    if n_proc > 1:
        pool = mp.Pool(n_proc)
        parout = []
        for parii in range(0, n_months):
            r = pool.apply_async(par_loop_month, args=(parii, month_list[parii], 
                                                   dn_nemo_data, fn_nemo_domain,
                                               dn_en4, dn_out, nemo_file_suffix,
                                               en4_file_prefix, n_obs_levels,
                                               month_season_dict, surface_def,
                                               bottom_def, dist_crit, run_name,
                                               regional_masks, region_names,
                                               depth_bins))
            parout.append(r)
        pool.close()
        pool.join()
        
        results = [p.get() for p in parout]
        
        file_list_sta = np.array([p[1] for p in results if p is not None])
        file_list_ext = np.array([p[2] for p in results if p is not None])
        parii_list = np.array( [p[0] for p in results if p is not None] )
        sort_ind = np.argsort(parii_list)
        parii_list = parii_list[sort_ind]
        file_list_sta = file_list_sta[sort_ind]
        file_list_ext = file_list_ext[sort_ind]
    else:
        file_list_sta = []
        file_list_ext = []
        for ii in range(0,n_months):
            out = par_loop_month(ii, month_list[ii], dn_nemo_data, fn_nemo_domain,
                                 dn_en4, dn_out, nemo_file_suffix, en4_file_prefix,
                                 n_obs_levels, month_season_dict, surface_def, 
                                 bottom_def, dist_crit, run_name, regional_masks,
                                 region_names, depth_bins)
            if out is not None:
                file_list_sta.append(out[1])
                file_list_ext.append(out[2])
    
    print('CONCATENATING OUTPUT FILES')
    
    # # Concatenate monthly output files into one file
    fn_stats = 'en4_stats_profiles_{0}.nc'.format(run_name)
    fn_stats = os.path.join(dn_out, fn_stats)
    all_stats = [xr.open_dataset(ff, chunks={}) for ff in file_list_sta]
    all_stats = xr.concat(all_stats, dim='profile')
    write_ds_to_file(all_stats, fn_stats)
    all_stats.close()
    
    for ff in range(0,len(file_list_sta)):
        os.remove(file_list_sta[ff])
    
    fn_ext = 'en4_extracted_profiles_{0}.nc'.format(run_name)
    fn_ext = os.path.join(dn_out, fn_ext)
    all_extracted = [xr.open_dataset(ff, chunks={}) for ff in file_list_ext]
    all_extracted = xr.concat(all_extracted, dim='profile')
    write_ds_to_file(all_extracted, fn_ext)
    all_extracted.close()
    for ff in range(0,len(file_list_ext)):
        os.remove(file_list_ext[ff])
        
    print('DOING REGIONAL ANALYSIS')
    
    n_regions = len(regional_masks) + 1
    n_regions=5
    
    n_seasons=5
    bin_widths = depth_bins[1:] - depth_bins[:-1]
    ref_depth = depth_bins[:-1] + .5*bin_widths
    n_ref_depth = len(ref_depth)
    
    if n_proc>1:
        pool = mp.Pool(n_proc)
        parout = []
        for rr in range(0, n_regions):
            for ss in range(0,n_seasons):
                r = pool.apply_async(par_loop_regional, args=(rr, ss, fn_stats))
                parout.append(r)
        pool.close()
        pool.join()
    
        results = [p.get() for p in parout]
        rr_list = [p[0] for p in results]
        ss_list = [p[1] for p in results]
        mean_list = [p[2] for p in results]
    else:
        rr_list= []
        ss_list = []
        mean_list=[]
        for rr in range(0,n_regions):
            for ss in range(0,n_seasons):
                out = par_loop_regional(rr, ss, fn_stats)
                rr_list.append(rr)
                ss_list.append(ss)
                mean_list.append(out[2])
                
        
    region_names = region_names.copy()
    region_names.append('whole_domain')
    
    regional_stats = xr.Dataset(coords = dict(
                                    region = ('region', region_names),
                                    bin_depth = ('bin_depth', ref_depth)),
                                data_vars = dict(
                                    prof_error_tem = (['region','season','bin_depth'], np.zeros((n_regions, n_seasons, n_ref_depth))*np.nan),
                                    prof_error_sal = (['region','season','bin_depth'], np.zeros((n_regions, n_seasons, n_ref_depth))*np.nan),
                                    prof_mean_mod_tem = (['region','season','bin_depth'], np.zeros((n_regions, n_seasons, n_ref_depth))*np.nan),
                                    prof_mean_mod_sal = (['region','season','bin_depth'], np.zeros((n_regions, n_seasons, n_ref_depth))*np.nan),
                                    prof_mean_obs_tem = (['region','season','bin_depth'], np.zeros((n_regions, n_seasons, n_ref_depth))*np.nan),
                                    prof_mean_obs_sal = (['region','season','bin_depth'], np.zeros((n_regions, n_seasons, n_ref_depth))*np.nan)))
    
    n_out = len(rr_list)
    for ii in range(0,n_out):
        rr = rr_list[ii]
        ss = ss_list[ii]
        mn = mean_list[ii]
        if mn is None:
            continue
        regional_stats.prof_error_tem[rr,ss] = mn.error_tem_binned
        regional_stats.prof_error_sal[rr,ss] = mn.error_sal_binned
        regional_stats.prof_mean_mod_tem[rr,ss] = mn.mod_tem_binned
        regional_stats.prof_mean_mod_sal[rr,ss] = mn.mod_sal_binned
        regional_stats.prof_mean_obs_tem[rr,ss] = mn.obs_tem_binned
        regional_stats.prof_mean_obs_sal[rr,ss] = mn.obs_sal_binned
        
    fn_region = 'en4_regional_stats_{0}.nc'.format(run_name)
    fn_region = os.path.join(dn_out, fn_region)
    write_ds_to_file(regional_stats, fn_region)


class en4_stats_radius_means():
    def __init__(self, fn_profile_stats, fn_out, grid_lon, grid_lat, 
                 radius=25):
        
        stats_profile = xr.open_mfdataset(fn_profile_stats, chunks={'profile':10000})
        seasons = ['Annual','DJF','MAM','JJA','SON']
        n_seasons = len(seasons)
        
        lon2, lat2 = np.meshgrid(grid_lon, grid_lat)
        
        lon = lon2.flatten()
        lat = lat2.flatten()
        
        surf_error_tem = np.zeros((n_seasons,len(lon)))
        surf_error_sal = np.zeros((n_seasons,len(lon)))
        surf_tem_N = np.zeros((n_seasons,len(lon)))
        surf_sal_N = np.zeros((n_seasons,len(lon)))
        
        bott_error_tem = np.zeros((n_seasons,len(lon)))
        bott_error_sal = np.zeros((n_seasons,len(lon)))
        bott_tem_N = np.zeros((n_seasons,len(lon)))
        bott_sal_N = np.zeros((n_seasons,len(lon)))
        
        for season in range(1,5):
            ind_season = stats_profile.season == season
            tmp = stats_profile.isel(profile=ind_season)
            tmp = tmp[['surf_error_tem', 'surf_error_sal','bott_error_tem','bott_error_sal']]
            tmp.load()
            
            # Remove outliers
            std = tmp.std(skipna = True)
            tmp['surf_error_tem'] = xr.where(uf.fabs(tmp['surf_error_tem']) > 5*std.surf_error_tem, np.nan, tmp['surf_error_tem'] )
            tmp['bott_error_tem'] = xr.where(uf.fabs(tmp['bott_error_tem']) > 5*std.surf_error_tem, np.nan, tmp['surf_error_tem'] )
            tmp['surf_error_sal'] = xr.where(uf.fabs(tmp['surf_error_sal']) > 5*std.surf_error_tem, np.nan, tmp['surf_error_sal'] )
            tmp['bott_error_sal'] = xr.where(uf.fabs(tmp['bott_error_sal']) > 5*std.surf_error_tem, np.nan, tmp['bott_error_sal'] )
            
            ind = coastgu.subset_indices_by_distance_BT(tmp.longitude, tmp.latitude, 
                                                lon, lat, radius=radius)
            
            tem = [tmp.surf_error_tem.isel(profile=ii).values for ii in ind]
            sal = [tmp.surf_error_sal.isel(profile=ii).values for ii in ind]
            surf_error_tem[season] = [np.nanmean(temii) for temii in tem]
            surf_error_sal[season] = [np.nanmean(salii) for salii in sal]
            surf_tem_N[season] = [np.sum( ~np.isnan(temii) ) for temii in tem]
            surf_sal_N[season] = [np.sum( ~np.isnan(salii) ) for salii in sal]
            
            tem = [tmp.bott_error_tem.isel(profile=ii).values for ii in ind]
            sal = [tmp.bott_error_sal.isel(profile=ii).values for ii in ind]
            bott_error_tem[season] = [np.nanmean(temii) for temii in tem]
            bott_error_sal[season] = [np.nanmean(salii) for salii in sal]
            bott_tem_N[season] = [np.sum( ~np.isnan(temii) ) for temii in tem]
            bott_sal_N[season] = [np.sum( ~np.isnan(salii) ) for salii in sal]
                
        ds = xr.Dataset(coords = dict(
                            longitude = ('location',lon),
                            latitude = ('location', lat),
                            season = ('season', seasons)),
                        data_vars = dict(
                            surf_error_tem = (['season','location'], surf_error_tem),
                            surf_error_sal = (['season','location'], surf_error_sal),
                            surf_tem_N = (['season','location'], surf_tem_N),
                            surf_sal_N = (['season','location'], surf_sal_N),
                            bott_error_tem = (['season','location'], bott_error_tem),
                            bott_error_sal = (['season','location'], bott_error_sal),
                            bott_tem_N = (['season','location'], bott_tem_N),
                            bott_sal_N = (['season','location'], bott_sal_N)))
        ds.to_netcdf(fn_out)

class plot_ts_radius_means_single_cfg():
    def __init__(self, fn_stats, dn_out, run_name, file_type='.png', min_N=1):
        
        stats = xr.open_mfdataset(fn_stats, chunks={})
        
        #Loop over seasons
        seasons = ['All','DJF','MAM','JJA','SON']
        lonmax = np.nanmax(stats.longitude)
        lonmin = np.nanmin(stats.longitude)
        latmax = np.nanmax(stats.latitude)
        latmin = np.nanmin(stats.latitude)
        lonbounds = [lonmin-1, lonmax+1]
        latbounds = [latmin-1, latmax+1]
        
        stats['surf_error_tem'] = xr.where(stats.surf_tem_N<min_N, np.nan, stats['surf_error_tem'])
        stats['surf_error_sal'] = xr.where(stats.surf_sal_N<min_N, np.nan, stats['surf_error_sal'])
        stats['bott_error_tem'] = xr.where(stats.surf_tem_N<min_N, np.nan, stats['bott_error_tem'])
        stats['bott_error_sal'] = xr.where(stats.surf_sal_N<min_N, np.nan, stats['bott_error_sal'])
        
        for ss in range(1, 5):

            stats_tmp = stats.isel(season=ss)   
        
            # Surface TEMPERATURE
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp.longitude, stats_tmp.latitude, c=stats_tmp.surf_error_tem, 
                vmin=-1.5, vmax=1.5, linewidths=0, cmap='seismic',s=2)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SST Anom. (degC) | {0} | {1} - EN4'.format(seasons[ss], run_name), fontsize=9)
            fn_out = 'en4_radius_means_surf_error_tem_{0}_{1}{2}'.format(seasons[ss], run_name, file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
            
            #Surface SALINITY
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp.longitude, stats_tmp.latitude, c=stats_tmp.surf_error_sal, 
                vmin=-1.5, vmax=1.5, linewidths=0, cmap='seismic', s=2)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SSS Anom. (PSU) | {0} | {1} - EN4'.format(seasons[ss], run_name), fontsize=9)
            fn_out = 'en4_radius_means_surf_error_sal_{0}_{1}{2}'.format(seasons[ss], run_name, file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
            
            # Bottom TEMPERATURE
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp.longitude, stats_tmp.latitude, c=stats_tmp.bott_error_tem, 
                vmin=-1.5, vmax=1.5, linewidths=0, cmap='seismic', s=2)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SBT Anom. (degC) | {0} | {1} - EN4'.format(seasons[ss], run_name), fontsize=9)
            fn_out = 'en4_radius_means_bott_error_tem_{0}_{1}{2}'.format(seasons[ss], run_name, file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
            
            # Bottom SALINITY
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp.longitude, stats_tmp.latitude, c=stats_tmp.bott_error_sal, 
                vmin=-1.5, vmax=1.5, linewidths=0, cmap='seismic', s=2)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SBS Anom. (PSU) | {0} | {1} - EN4'.format(seasons[ss], run_name), fontsize=9)
            fn_out = 'en4_radius_means_bott_error_sal_{0}_{1}{2}'.format(seasons[ss], run_name, file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
        return
    
class plot_ts_radius_means_comparison():
    def __init__(self, fn_stats1, fn_stats2, dn_out, run_name, file_type='.png', min_N=1):
        
        stats1 = xr.open_mfdataset(fn_stats1, chunks={})
        stats2 = xr.open_mfdataset(fn_stats2, chunks={})
        
        #Loop over seasons
        seasons = ['All','DJF','MAM','JJA','SON']
        lonmax = np.nanmax(stats1.longitude)
        lonmin = np.nanmin(stats1.longitude)
        latmax = np.nanmax(stats1.latitude)
        latmin = np.nanmin(stats1.latitude)
        lonbounds = [lonmin-1, lonmax+1]
        latbounds = [latmin-1, latmax+1]
        
        stats1['surf_error_tem'] = xr.where(stats1.surf_tem_N<min_N, np.nan, uf.fabs(stats1['surf_error_tem']))
        stats1['surf_error_sal'] = xr.where(stats1.surf_sal_N<min_N, np.nan, uf.fabs(stats1['surf_error_sal']))
        stats1['bott_error_tem'] = xr.where(stats1.surf_tem_N<min_N, np.nan, uf.fabs(stats1['bott_error_tem']))
        stats1['bott_error_sal'] = xr.where(stats1.surf_sal_N<min_N, np.nan, uf.fabs(stats1['bott_error_sal']))
        
        stats2['surf_error_tem'] = xr.where(stats2.surf_tem_N<min_N, np.nan, uf.fabs(stats2['surf_error_tem']))
        stats2['surf_error_sal'] = xr.where(stats2.surf_sal_N<min_N, np.nan, uf.fabs(stats2['surf_error_sal']))
        stats2['bott_error_tem'] = xr.where(stats2.surf_tem_N<min_N, np.nan, uf.fabs(stats2['bott_error_tem']))
        stats2['bott_error_sal'] = xr.where(stats2.surf_sal_N<min_N, np.nan, uf.fabs(stats2['bott_error_sal']))
        
        for ss in range(1, 5):

            stats_tmp1 = stats1.isel(season=ss)   
            stats_tmp2 = stats2.isel(season=ss)   
        
            # Surface TEMPERATURE
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp1.longitude, stats_tmp1.latitude, c=stats_tmp2.surf_error_tem - stats_tmp1.surf_error_tem, 
                vmin=-.25, vmax=.25, linewidths=0, cmap='PiYG',s=2)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SST Abs. Anom. Difference (degC) | {0} | {1} - {2}'.format(seasons[ss], run_name[1], run_name[0]), fontsize=9)
            fn_out = 'en4_radius_means_surf_error_tem_{0}_{1}_{2}{3}'.format(seasons[ss], run_name[0], run_name[1], file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
            
            #Surface SALINITY
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp1.longitude, stats_tmp1.latitude, c=stats_tmp2.surf_error_sal - stats_tmp1.surf_error_sal, 
                vmin=-.25, vmax=.25, linewidths=0, cmap='PiYG', s=2)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SSS Abs. Anom. Difference (PSU) | {0} | {1} - {2}'.format(seasons[ss], run_name[1], run_name[0]), fontsize=9)
            fn_out = 'en4_radius_means_surf_error_sal_{0}_{1}_{2}{3}'.format(seasons[ss], run_name[0], run_name[1], file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
            
            # Bottom TEMPERATURE
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp1.longitude, stats_tmp1.latitude, c=stats_tmp2.bott_error_tem - stats_tmp1.bott_error_tem, 
                vmin=-.25, vmax=.25, linewidths=0, cmap='PiYG', s=2)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SBT Abs. Anom. Difference (degC) | {0} | {1} - {2}'.format(seasons[ss], run_name[1], run_name[0]), fontsize=9)
            fn_out = 'en4_radius_means_bott_error_tem_{0}_{1}_{2}{3}'.format(seasons[ss], run_name[0], run_name[1], file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
            
            # Bottom SALINITY
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp1.longitude, stats_tmp1.latitude, c=stats_tmp2.bott_error_sal - stats_tmp1.bott_error_sal, 
                vmin=-.25, vmax=.25, linewidths=0, cmap='PiYG', s=2)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SBS Abs. Anom. Difference (PSU) | {0} | {1} - {2}'.format(seasons[ss], run_name[1], run_name[0]), fontsize=9)
            fn_out = 'en4_radius_means_bott_error_sal_{0}_{1}_{2}{3}'.format(seasons[ss], run_name[0], run_name[1], file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
        
        return

class plot_ts_monthly_single_cfg():
    
    def __init__(self, fn_profile_stats, dn_out, run_name, file_type='.png'):

        stats = xr.open_mfdataset(fn_profile_stats, chunks={})
        
        #Loop over seasons
        seasons = ['All','DJF','MAM','JJA','SON']
        lonmax = np.nanmax(stats.longitude)
        lonmin = np.nanmin(stats.longitude)
        latmax = np.nanmax(stats.latitude)
        latmin = np.nanmin(stats.latitude)
        lonbounds = [lonmin-1, lonmax+1]
        latbounds = [latmin-1, latmax+1]
        
        for ss in range(0, 5):
        
            if ss>0:
                ind_season = stats.season==ss
                stats_tmp = stats.isel(profile=ind_season)    
            else:
                stats_tmp = stats
        
            # Surface TEMPERATURE
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp.longitude, stats_tmp.latitude, c=stats_tmp.surf_error_tem, 
                vmin=-1.5, vmax=1.5, linewidths=0, zorder=100, cmap='seismic',s=1)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SST Anom. (degC) | {0} | {1} - EN4'.format(seasons[ss], run_name), fontsize=9)
            fn_out = 'en4_surf_error_tem_{0}_{1}{2}'.format(seasons[ss], run_name, file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
            
            #Surface SALINITY
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp.longitude, stats_tmp.latitude, c=stats_tmp.surf_error_sal, 
                vmin=-1.5, vmax=1.5, linewidths=0, zorder=100, cmap='seismic', s=1)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SSS Anom. (PSU) | {0} | {1} - EN4'.format(seasons[ss], run_name), fontsize=9)
            fn_out = 'en4_surf_error_sal_{0}_{1}{2}'.format(seasons[ss], run_name, file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
            
            # Bottom TEMPERATURE
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp.longitude, stats_tmp.latitude, c=stats_tmp.bott_error_tem, 
                vmin=-1.5, vmax=1.5, linewidths=0, zorder=100, cmap='seismic', s=1)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SBT Anom. (degC) | {0} | {1} - EN4'.format(seasons[ss], run_name), fontsize=9)
            fn_out = 'en4_bott_error_tem_{0}_{1}{2}'.format(seasons[ss], run_name, file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
            
            # Bottom SALINITY
            f,a = pu.create_geo_axes(lonbounds, latbounds)
            sca = a.scatter(stats_tmp.longitude, stats_tmp.latitude, c=stats_tmp.bott_error_sal, 
                vmin=-1.5, vmax=1.5, linewidths=0, zorder=100, cmap='seismic', s=1)
            f.colorbar(sca)
            a.set_title('Monthly EN4 SBS Anom. (PSU) | {0} | {1} - EN4'.format(seasons[ss], run_name), fontsize=9)
            fn_out = 'en4_bott_error_sal_{0}_{1}{2}'.format(seasons[ss], run_name, file_type)
            fn_out = os.path.join(dn_out, fn_out)
            f.savefig(fn_out)
    
    
class plot_ts_monthly_multi_cfg():
    def __init__(self, fn_regional_stats, dn_out, run_name, file_type='.png'):
        
        stats_list = [xr.open_dataset(ff, chunks={}) for ff in fn_regional_stats]
    
        # For titles
        region_names = ['North Sea','Outer Shelf','Norwegian Trench','English Channel','Whole Domain']
        # For file names
        region_abbrev = ['northsea','outershelf','nortrench','engchannel','wholedomain']
        
        season_names = ['Annual','DJF','MAM','JJA','SON']
        legend = run_name
        n_regions = len(region_names)
        n_seasons = len(season_names)
        for rr in range(0,n_regions):
            for ss in range(1,n_seasons):
                tem_list = [tmp.prof_error_tem.isel( region=rr, season=ss, depth=np.arange(0,30) ) for tmp in stats_list]
                sal_list = [tmp.prof_error_sal.isel( region=rr, season=ss, depth=np.arange(0,30) ) for tmp in stats_list]
                
                title_tmp = '$\Delta T$ (degC) | {0} | {1}'.format(region_names[rr], season_names[ss])
                fn_out = 'prof_error_tem_{0}_{1}{2}'.format(season_names[ss], region_abbrev[rr], file_type)
                fn_out = os.path.join(dn_out, fn_out)
                f,a = self.plot_profile_centred(tem_list[0].depth, tem_list,
                              title = title_tmp, legend_names = legend)
                print("  >>>>>  Saving: " + fn_out)
                f.savefig(fn_out)
                plt.close()
                
                title_tmp = '$\Delta S$ (PSU) |' + region_names[rr] +' | '+season_names[ss]
                fn_out = 'prof_error_sal_{0}_{1}{2}'.format(season_names[ss], region_abbrev[rr], file_type)
                fn_out = os.path.join(dn_out, fn_out)
                f,a = self.plot_profile_centred(sal_list[0].depth, sal_list,
                             title = title_tmp,legend_names = legend)
                print("  >>>>>  Saving: " + fn_out)
                f.savefig(fn_out)
                plt.close()
    
                
    def plot_profile_centred(self, depth, variables, title="", legend_names= {} ):
    
        fig = plt.figure(figsize=(3.5,7))
        ax = plt.subplot(111)
        
        if type(variables) is not list:
            variables = [variables]
    
        xmax = 0
        for vv in variables:
            xmax = np.max([xmax, np.nanmax(np.abs(vv))])
            ax.plot(savgol_filter(vv.squeeze(),5,2), depth.squeeze())
            
        plt.xlim(-xmax-0.05*xmax, xmax+0.05*xmax)
        ymax = np.nanmax(np.abs(depth))
        plt.plot([0,0],[-1e7,1e7], linestyle='--',linewidth=1,color='k')
        plt.ylim(0,ymax)
        plt.gca().invert_yaxis()
        plt.ylabel('Depth (m)')
        plt.grid()
        plt.legend(legend_names, fontsize=10)
        plt.title(title, fontsize=8)
        return fig, ax
                
    def plot_profile(self, depth, variables, title, fn_out, legend_names= {} ):
    
        fig = plt.figure(figsize=(3.5,7))
        ax = plt.subplot(111)
        
        if type(variables) is not list:
            variables = [variables]
    
        for vv in variables:
            ax.plot(vv.squeeze(), depth.squeeze())
        plt.gca().invert_yaxis()
        plt.ylabel('Depth (m)')
        plt.grid()
        plt.legend(legend_names)
        plt.title(title, fontsize=12)
        print("  >>>>>  Saving: " + fn_out)
        plt.savefig(fn_out)
        plt.close()
        return fig, ax
    
# class analyse_ts_monthly_en4():
    
#     def __init__(self, dn_nemo_data, fn_nemo_domain, dn_en4, dn_output, 
#                   start_month, end_month, run_name,
#                   nemo_file_suffix = '01_monthly_grid_T',
#                   en4_file_prefix = 'EN.4.2.1.f.profiles.g10.',
#                   regional_masks=[],
#                   surface_def = 5, bottom_def = 10, mld_ref_depth = 5,
#                   mld_threshold = 0.02, dist_crit=5):

#         # Define a counter which keeps track of the current month
#         current_month = start_month
#         n_months = (end_month.year - start_month.year)*12 + \
#                         (end_month.month - start_month.month) + 1
                        
#         # Make an initial NEMO file name for month 0 and read the data
#         fn_nemo_month = make_nemo_filename(dn_nemo_data, current_month, nemo_file_suffix)
#         mod_month = read_monthly_model_nemo(fn_nemo_month, fn_nemo_domain)
        
#         # Define a log file to output progress
#         print(' *Profile analysis starting.*')
        
#         # Number of levels in EN4 data
#         n_obs_levels = 400
        
#         # Define regional masks for regional averaging / profiles
#         n_r = mod_month.dims['y_dim']
#         n_c = mod_month.dims['x_dim']
#         regional_masks.append(np.ones((n_r, n_c)))
#         n_regions = len(regional_masks) # Number of regions
        
#         # Define seasons for seasonal averaging/collation
#         month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
#                               7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
#         season_names = ['whole_year','DJF','MAM','JJA','SON']
#         n_seasons = len(season_names) # 0 = all year, 1-4 = winter -> autumn
        
#         # Define depth bins for error averaging with depth
#         ref_depth_bins = np.concatenate([ np.arange(-0.001,200,5) ])
#         bin_widths = ref_depth_bins[1:] - ref_depth_bins[:-1]
#         ref_depth = ref_depth_bins[:-1] + .5*bin_widths
#         n_ref_depth_bins = len(ref_depth_bins)
#         n_ref_depth = len(ref_depth)
        
#         # Define arrays which will contain regional and seasonal mean profile 
#         # errors
#         prof_error_tem = np.zeros((n_regions, n_seasons, n_ref_depth))
#         prof_error_sal = np.zeros((n_regions, n_seasons, n_ref_depth))
#         prof_error_tem_N = np.zeros((n_regions, n_seasons, n_ref_depth))
#         prof_error_sal_N = np.zeros((n_regions, n_seasons, n_ref_depth))
        
#         profiles_analysed = 0 # Counter for keeping track of all profiles analyzed
#         profiles_saved = 0 # Counter for keeping track of all profiles saved to file
        
#         # Output Datasets -- For final stats and to write to file 
#         tmp_file_names_ext = []
#         tmp_file_names_sta = []
        
#         # ----------------------------------------------------
#         # ANALYSIS: Load in month by month and interpolate profile by profile
        
#         for month_ii in range(0,n_months):
            
#             print('Loading month of data: ' + str(current_month))
            
#             # Make filenames for current month and read NEMO/EN4
#             # If this files (e.g. file does not exist), skip to next month
#             try:
#                 fn_nemo_month = make_nemo_filename(dn_nemo_data, current_month, nemo_file_suffix)
#                 fn_en4_month = make_en4_filename(dn_en4, current_month, en4_file_prefix)
#                 mod_month = read_monthly_model_nemo(fn_nemo_month, fn_nemo_domain)
#                 obs_month = read_monthly_profile_en4(fn_en4_month)
#             except:
#                 current_month = current_month + relativedelta(months=+1)
            
#                 print('       !!!Problem with read: Not analyzed ')
#                 continue
            
#             # ----------------------------------------------------
#             # Init 2) Use only observations that are within model domain
#             lonmax = np.nanmax(mod_month['longitude'])
#             lonmin = np.nanmin(mod_month['longitude'])
#             latmax = np.nanmax(mod_month['latitude'])
#             latmin = np.nanmin(mod_month['latitude'])
#             ind = coast.general_utils.subset_indices_lonlat_box(obs_month['longitude'], 
#                                                                 obs_month['latitude'],
#                                                                 lonmin, lonmax, 
#                                                                 latmin, latmax)[0]
#             obs_month = obs_month.isel(profile=ind)
#             obs_month = obs_month.load()
#             print(obs_month)
            
#             # ----------------------------------------------------
#             # Init 4) Get model indices (space and time) corresponding to observations
#             ind2D = coastgu.nearest_indices_2D(mod_month['longitude'], 
#                                                 mod_month['latitude'],
#                                                 obs_month['longitude'], 
#                                                 obs_month['latitude'], 
#                                                 mask=mod_month.landmask)
    
#             n_prof = len(obs_month.time) # Number of profiles in this month
#             mod_profiles = mod_month.isel(x_dim=ind2D[0], y_dim=ind2D[1])
#             mod_profiles = mod_profiles.rename({'dim_0':'profile'})
            
#             # Define monthly variable arrays for interpolated data
#             mod_tem = np.zeros((n_prof , n_obs_levels))*np.nan
#             obs_tem = np.zeros((n_prof, n_obs_levels))*np.nan
#             mod_sal = np.zeros((n_prof, n_obs_levels))*np.nan
#             obs_sal = np.zeros((n_prof, n_obs_levels))*np.nan
#             mod_rho = np.zeros((n_prof, n_obs_levels))*np.nan
#             obs_rho = np.zeros((n_prof, n_obs_levels))*np.nan
#             mod_s0 = np.zeros((n_prof, n_obs_levels))*np.nan
#             obs_s0 = np.zeros((n_prof, n_obs_levels))*np.nan
#             obs_z = np.zeros((n_prof, n_obs_levels))*np.nan
            
#             # Determine the current season integer
#             current_season = month_season_dict[current_month.month]
            
#             # Loop over profiles, interpolate model to obs depths and store in
#             # monthly arrays
#             ind_prof_use = []
#             fail_reason = np.zeros(n_prof) # Debugging variable
            
            
#             for prof in range(0,n_prof):
                
#                 # Select the current profile
#                 mod_profile = mod_profiles.isel(profile = prof)
#                 obs_profile = obs_month.isel(profile = prof)
                
#                 # If the nearest neighbour interpolation is bad, then skip the 
#                 # vertical interpolation -> keep profile as nans in monthly array
#                 if all(np.isnan(mod_profile.temperature)):
#                     profiles_analysed += 1
#                     fail_reason[prof] = 1
#                     continue
                
#                 # Check that model point is within threshold distance of obs
#                 # If not, skip vertical interpolation -> keep profile as nans
#                 interp_dist = coastgu.calculate_haversine_distance(
#                                                       obs_profile.longitude, 
#                                                       obs_profile.latitude, 
#                                                       mod_profile.longitude, 
#                                                       mod_profile.latitude)
#                 if interp_dist > dist_crit:
#                     profiles_analysed+=1
#                     fail_reason[prof] = 2
#                     continue
                
#                 # Use bottom_level to mask dry depths
#                 bl = mod_profile.bottom_level.squeeze().values
#                 mod_profile = mod_profile.isel(z_dim=range(0,bl))
                
#                 #
#                 # Interpolate model to obs depths using a linear interp
#                 #
#                 obs_profile = obs_profile.rename({'z_dim':'depth'})
#                 obs_profile = obs_profile.set_coords('depth')
#                 mod_profile = mod_profile.rename({'z_dim':'depth_0'})
                
#                 # If interpolation fails for some reason, skip to next iteration
#                 try:
#                     mod_profile_int = mod_profile.interp(depth_0 = obs_profile.depth.values)
#                 except:
#                     profiles_analysed+=1
#                     fail_reason[prof] = 3
#                     continue
                
#                 # ----------------------------------------------------
#                 # Analysis 2) Calculate Density per Profile using GSW
                
#                 # Calculate Density
#                 ap_obs = gsw.p_from_z( -obs_profile.depth, obs_profile.latitude )
#                 ap_mod = gsw.p_from_z( -obs_profile.depth, mod_profile_int.latitude )
#                 # Absolute Salinity            
#                 sa_obs = gsw.SA_from_SP( obs_profile.salinity, ap_obs, 
#                                         obs_profile.longitude, 
#                                         obs_profile.latitude )
#                 sa_mod = gsw.SA_from_SP( mod_profile_int.salinity, ap_mod, 
#                                         mod_profile_int.longitude, 
#                                         mod_profile_int.latitude )
#                 # Conservative Temperature
#                 ct_obs = gsw.CT_from_pt( sa_obs, obs_profile.temperature ) 
#                 ct_mod = gsw.CT_from_pt( sa_mod, mod_profile_int.temperature ) 
                
#                 # In-situ density
#                 obs_rho_tmp = gsw.rho( sa_obs, ct_obs, ap_obs )
#                 mod_rho_tmp = gsw.rho( sa_mod, ct_mod, ap_mod ) 
                
#                 # Potential Density
#                 obs_s0_tmp = gsw.sigma0(sa_obs, ct_obs)
#                 mod_s0_tmp = gsw.sigma0(sa_mod, ct_mod)
                
#                 # Assign monthly array
#                 mod_tem[prof] = mod_profile_int.temperature.values
#                 obs_tem[prof] = obs_profile.temperature.values
#                 mod_sal[prof] = mod_profile_int.salinity.values
#                 obs_sal[prof] = obs_profile.salinity.values
#                 mod_rho[prof] = mod_rho_tmp
#                 obs_rho[prof] = obs_rho_tmp
#                 mod_s0[prof] = mod_s0_tmp
#                 obs_s0[prof] = obs_s0_tmp
#                 obs_z[prof] = obs_profile.depth
                
#                 # If got to this point then keep the profile
#                 ind_prof_use.append(prof)
#                 profiles_analysed+=1
                
#             print('       Interpolated Profiles.')
#             # Find the union of masks for each variable
#             mask_tem = np.logical_or(np.isnan(mod_tem), np.isnan(obs_tem))
#             mask_sal = np.logical_or(np.isnan(mod_sal), np.isnan(obs_sal))
#             mask_rho = np.logical_or(np.isnan(mod_rho), np.isnan(obs_rho))
            
#             mod_tem[mask_tem] = np.nan
#             obs_tem[mask_tem] = np.nan
#             mod_sal[mask_sal] = np.nan
#             obs_sal[mask_sal] = np.nan
            
#             # Monthly stats xarray dataset - for file output and easy indexing
#             season_save_vector = np.ones( n_prof, dtype=int )*current_season
#             data = xr.Dataset(coords = dict(
#                                       longitude=(["profile"], obs_month.longitude),
#                                       latitude=(["profile"], obs_month.latitude),
#                                       time=(["profile"], obs_month.time),
#                                       level=(['level'], np.arange(0,400))),
#                                   data_vars = dict(
#                                       mod_tem = (['profile','level'], mod_tem),
#                                       obs_tem = (['profile','level'], obs_tem),
#                                       mod_sal = (['profile','level'], mod_sal),
#                                       obs_sal = (['profile','level'], obs_sal),
#                                       mod_rho = (['profile','level'], mod_rho),
#                                       obs_rho = (['profile','level'], obs_rho),
#                                       mod_s0 = (['profile', 'level'], mod_s0),
#                                       obs_s0 = (['profile', 'level'], obs_s0),
#                                       mask_tem = (['profile','level'], mask_tem),
#                                       mask_sal = (['profile','level'], mask_sal),
#                                       mask_rho = (['profile','level'], mask_rho),
#                                       obs_z = (['profile','level'], obs_z)))
            
#             analysis = xr.Dataset(coords = dict(
#                                       longitude=(["profile"], obs_month.longitude),
#                                       latitude=(["profile"], obs_month.latitude),
#                                       time=(["profile"], obs_month.time),
#                                       level=(['level'], np.arange(0,400))),
#                               data_vars = dict(
#                                       season = (['profile'], season_save_vector),
#                                       obs_z = (['profile','level'], obs_z)))
            
#             # Keep the profiles we want to keep
#             n_prof_use = len(ind_prof_use)
#             analysis = analysis.isel(profile = ind_prof_use)
#             data = data.isel(profile = ind_prof_use)
            
#             # ----------------------------------------------------
#             # Analysis 3) All Anomalies
#             # Errors at all depths
#             analysis["error_tem"] = data.mod_tem - data.obs_tem
#             analysis["error_sal"] = data.mod_sal - data.obs_sal
            
#             # Absolute errors at all depths
#             analysis["abs_error_tem"] = np.abs(analysis.error_tem)
#             analysis["abs_error_sal"] = np.abs(analysis.error_sal)
            
#             # ----------------------------------------------------
#             # Analysis 7) Whole profile stats
#             # Mean errors across depths
            
#             analysis['me_tem'] = ('profile', np.nanmean(analysis.error_tem, axis=1))
#             analysis['me_sal'] = ('profile', np.nanmean(analysis.error_sal, axis=1))
            
#             # Mean absolute errors across depths
#             analysis['mae_tem'] = (['profile'], 
#                                     np.nanmean(analysis.abs_error_tem, axis=1))
#             analysis['mae_sal'] = (['profile'], 
#                                     np.nanmean(analysis.abs_error_tem, axis=1))
            
#             print('       Basic errors done. ')
            
#             # ----------------------------------------------------
#             # Analysis 4) Surface stats
#             # Get indices corresponding to surface depth
#             # Get averages over surface depths
#             analysis['surface_definition'] = surface_def
#             surface_ind = data.obs_z.values <= surface_def
            
#             mod_sal_tmp = np.array(data.mod_sal)
#             mod_tem_tmp = np.array(data.mod_tem)
#             obs_sal_tmp = np.array(data.obs_sal)
#             obs_tem_tmp = np.array(data.obs_tem)

#             mod_sal_tmp[~surface_ind] = np.nan
#             mod_tem_tmp[~surface_ind] = np.nan
#             obs_sal_tmp[~surface_ind] = np.nan
#             obs_tem_tmp[~surface_ind] = np.nan
                
#             # Average over surfacedepths
#             mod_surf_tem = np.nanmean(mod_tem_tmp, axis=1)
#             mod_surf_sal = np.nanmean(mod_sal_tmp, axis=1)
#             obs_surf_tem = np.nanmean(obs_tem_tmp, axis=1)
#             obs_surf_sal = np.nanmean(obs_sal_tmp, axis=1)
 
#             # Assign to output arrays
#             surf_error_tem =  mod_surf_tem - obs_surf_tem
#             surf_error_sal =  mod_surf_sal - obs_surf_sal
            
#             analysis['surf_error_tem'] = (['profile'], surf_error_tem)
#             analysis['surf_error_sal'] = (['profile'], surf_error_sal)
            
#             # ----------------------------------------------------
#             # Analysis 5) Bottom stats
#             # Estimate ocean depth as sum of e3t
#             analysis['bottom_definition'] = bottom_def
#             prof_bathy = mod_profiles.bathymetry.isel(profile=ind_prof_use)
#             percent_depth = bottom_def
#             # Get indices of bottom depths
#             bott_ind = data.obs_z.values >= (prof_bathy - percent_depth).values[:,None]
            
#             mod_sal_tmp = np.array(data.mod_sal)
#             mod_tem_tmp = np.array(data.mod_tem)
#             obs_sal_tmp = np.array(data.obs_sal)
#             obs_tem_tmp = np.array(data.obs_tem)
            
#             mod_sal_tmp[~bott_ind] = np.nan
#             mod_tem_tmp[~bott_ind] = np.nan
#             obs_sal_tmp[~bott_ind] = np.nan
#             obs_tem_tmp[~bott_ind] = np.nan
                
#             # Average over bottom depths
#             mod_bott_sal = np.nanmean(mod_sal_tmp, axis=1)
#             obs_bott_tem = np.nanmean(obs_tem_tmp, axis=1)
#             mod_bott_tem = np.nanmean(mod_tem_tmp, axis=1)
#             obs_bott_sal = np.nanmean(obs_sal_tmp, axis=1)
            
#             analysis['bott_error_tem'] = (['profile'], mod_bott_tem - obs_bott_tem)
#             analysis['bott_error_sal'] = (['profile'], mod_bott_sal - obs_bott_sal)
            
#             print('       Surface and bottom errors done. ')
            
#             # ----------------------------------------------------
#             # Analysis 6) Mixed Layer Depth - needs GSW EOS
#             # Absolute Pressure 
#             # surface_ind = ref_depth<= mld_reference_depth
#             # data_surf = data.isel(depth=surface_ind)
#             # reference_s0_obs = np.nanmean(data_surf.obs_s0, axis=1)
#             # reference_s0_mod = np.nanmean(data_surf.mod_s0, axis=1)
#             # s0_diff_obs = np.abs( (data.obs_s0.T - reference_s0_obs).T )
#             # s0_diff_mod = np.abs( (data.mod_s0.T - reference_s0_mod).T )
#             # ind_mld_obs = (s0_diff_obs > mld_threshold).argmax(dim='depth')
#             # ind_mld_mod = (s0_diff_mod > mld_threshold).argmax(dim='depth')
#             # mld_obs = ref_depth[ind_mld_obs].astype(float)
#             # mld_mod = ref_depth[ind_mld_mod].astype(float)
#             # mld_obs[mld_obs<mld_reference_depth] = np.nan
#             # mld_mod[mld_mod<mld_reference_depth] = np.nan
            
#             # analysis['mld_obs'] = (['profile'], mld_obs)
#             # analysis['mld_mod'] = (['profile'], mld_mod)
            
#             # with open(fn_log, 'a') as f:
#             #     f.write('       Mixed Layer Depth done. \n')
            
#             save_mask_ind = np.zeros((n_regions, n_prof_use), dtype=bool)
#             ind_season = [0, current_season] # 0 = whole year
#             ind2D_use[0] = ind2D[0][ind_prof_use]
                    
#             # Loop over profiles and put errors into depth bins
#             # < Currently a bit of a time bottleneck because of nested loops >
#             for prof in range(0, data.dims['profile']):
#                 data_profile = data.isel(profile = prof)
#                 ana_profile = analysis.isel(profile=prof)
                
#                 # Find mean error by depth bins for current profile
#                 error_tem_bin = spst.binned_statistic(data_profile.obs_z, 
#                                           ana_profile.error_tem, 'mean', 
#                                           ref_depth_bins)[0]
#                 error_sal_bin = spst.binned_statistic(data_profile.obs_z, 
#                                           ana_profile.error_sal, 'mean', 
#                                           ref_depth_bins)[0]
                
#                 # Loop over regional masks and determine if profile lies within
#                 for region_ii in range(0, n_regions):
#                     # Get mask
#                     mask = regional_masks[region_ii]
#                     # If index is True then profile is in region
#                     is_in_region = mask[ind2D[1][ind_prof_use][prof], 
#                                         ind2D[0][ind_prof_use][prof]]
#                     save_mask_ind[region_ii, prof] = is_in_region
                    
#                     if not is_in_region:
#                         continue
                    
#                     for ss in ind_season:
#                         # Only add depths which don't contain NaNs
#                         tem_notnan = ~np.isnan(error_tem_bin)
#                         sal_notnan = ~np.isnan(error_sal_bin)
#                         prof_error_tem[region_ii, ss, tem_notnan] += error_tem_bin[tem_notnan]
#                         prof_error_sal[region_ii, ss, sal_notnan] += error_sal_bin[sal_notnan]
#                         prof_error_tem_N[region_ii, ss, tem_notnan] += 1
#                         prof_error_sal_N[region_ii, ss, sal_notnan] += 1
                        
                      
#             # ----------------------------------------------------
#             # WRITE 1) Write monthly stats to file
            
#             # Postproc 1) Write data to file for each month - profiles stats
#             yy = str(current_month.year)
#             mm = str(current_month.month)
            
#             # Create a diension coordinate that increases monotonically for
#             # easy concatenation using xarray.open_mfdataset
#             profile_id = np.arange(profiles_saved, profiles_saved + len(ind_prof_use) )
#             profiles_saved = profiles_saved + len(ind_prof_use)
#             analysis['profile'] = ('profile', profile_id)
#             analysis['region_bool'] = (['region','profile'], save_mask_ind)
            
#             # Create temp monthly file and write to it
#             fn_tmp = 'en4_stats_by_profile_{0}{1}_{2}.nc'.format(yy,mm.zfill(2),run_name)
#             fn = os.path.join(dn_output, fn_tmp)
#             tmp_file_names_sta.append(fn)
#             write_ds_to_file(analysis, fn, mode='w', unlimited_dims='profile')
            
#             print('       File Written: ' + fn_tmp)
                
                
#             # ----------------------------------------------------
#             # WRITE 2) Write monthly extracted profiles to file
            
#             # Create a diension coordinate that increases monotonically for
#             # easy concatenation using xarray.open_mfdataset
#             data['profile'] = ('profile', profile_id)
#             data['region_bool'] = (['region','profile'], save_mask_ind)
            
#             # Create temp monthly file and write to it
#             fn_tmp = 'en4_extracted_profiles_{0}{1}_{2}.nc'.format(yy,mm.zfill(2), run_name)
#             fn = os.path.join(dn_output, fn_tmp)
#             tmp_file_names_ext.append(fn)
#             write_ds_to_file(data, fn, mode='w', unlimited_dims='profile')
            
#             print('       File Written: ' + fn_tmp)
            
#             # Onto next month
#             current_month = current_month + relativedelta(months=+1)
            
#             ####
#             ## END OF MONTH LOOP
#             ####
            
#         # Complete calculations of average profiles and regional statistics
#         prof_error_tem = prof_error_tem/prof_error_tem_N
#         prof_error_sal = prof_error_sal/prof_error_sal_N
         
#         # Postproc 1) Write data to file for each month - profiles stats
#         season_names = ['whole_year','DJF','MAM','JJA','SON']
#         stats_regional = xr.Dataset( 
#           data_vars = dict(
#               prof_error_tem = (['region','season','depth'], prof_error_tem),
#               prof_error_sal = (['region','season','depth'], prof_error_sal)),
#           coords = dict(
#               season_names = ('season', season_names),
#               depth = ('depth', ref_depth)))
         
#         fn_stats_region = 'en4_stats_regional_{0}.nc'.format(run_name)
#         fn = os.path.join(dn_output, fn_stats_region)
#         write_ds_to_file(stats_regional, fn, mode='w')
        
#         print('CONCATENATING OUTPUT FILES')
        
#         # Concatenate monthly output files into one file
#         fn_stats = 'en4_stats_profiles_{0}.nc'.format(run_name)
#         fn = os.path.join(dn_output, fn_stats)
#         all_stats = xr.open_mfdataset(tmp_file_names_sta, combine='by_coords', chunks={'profile':10000})
#         write_ds_to_file(all_stats, fn)
#         for ff in tmp_file_names_sta:
#             os.remove(ff)
        
#         fn_ext = 'en4_extracted_profiles_{0}.nc'.format(run_name)
#         fn = os.path.join(dn_output, fn_ext)
#         all_extracted = xr.open_mfdataset(tmp_file_names_ext,  combine='by_coords', chunks={'profile':10000})
#         write_ds_to_file(all_extracted, fn)
#         for ff in tmp_file_names_ext:
#             os.remove(ff)
