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
    ds = nemo.dataset[['temperature','salinity', 'e3t','depth_0', 'landmask','bathymetry']]
    return ds.squeeze()

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
    
 
class analyse_ts_monthly_en4():
    
    def __init__(self, dn_nemo_data, fn_nemo_domain, dn_en4, dn_output, 
                 start_month, end_month, run_name,
                 nemo_file_suffix = '01_monthly_grid_T',
                 en4_file_prefix = 'EN.4.2.1.f.profiles.g10.',
                 regional_masks=[],
                 surface_def = 5, bottom_def = 10, mld_ref_depth = 5,
                 mld_threshold = 0.02, dist_crit=5):

        # Define a counter which keeps track of the current month
        current_month = start_month
        n_months = (end_month.year - start_month.year)*12 + \
                        (end_month.month - start_month.month) + 1
                        
        # Make an initial NEMO file name for month 0 and read the data
        fn_nemo_month = make_nemo_filename(dn_nemo_data, current_month, nemo_file_suffix)
        mod_month = read_monthly_model_nemo(fn_nemo_month, fn_nemo_domain)
        
        # Define a log file to output progress
        print(' *Profile analysis starting.*')
        
        # Number of levels in EN4 data
        n_obs_levels = 400
        
        # Define regional masks for regional averaging / profiles
        n_r = mod_month.dims['y_dim']
        n_c = mod_month.dims['x_dim']
        print(n_r)
        regional_masks.append(np.ones((n_r, n_c)))
        n_regions = len(regional_masks) # Number of regions
        
        # Define seasons for seasonal averaging/collation
        month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                             7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
        season_names = ['whole_year','DJF','MAM','JJA','SON']
        n_seasons = len(season_names) # 0 = all year, 1-4 = winter -> autumn
        
        # Define depth bins for error averaging with depth
        ref_depth_bins = np.concatenate([ np.arange(-0.001,200,5) ])
        bin_widths = ref_depth_bins[1:] - ref_depth_bins[:-1]
        ref_depth = ref_depth_bins[:-1] + .5*bin_widths
        n_ref_depth_bins = len(ref_depth_bins)
        n_ref_depth = len(ref_depth)
        
        # Define arrays which will contain regional and seasonal mean profile 
        # errors
        prof_error_tem = np.zeros((n_regions, n_seasons, n_ref_depth))
        prof_error_sal = np.zeros((n_regions, n_seasons, n_ref_depth))
        prof_error_tem_N = np.zeros((n_regions, n_seasons, n_ref_depth))
        prof_error_sal_N = np.zeros((n_regions, n_seasons, n_ref_depth))
        
        profiles_analysed = 0 # Counter for keeping track of all profiles analyzed
        profiles_saved = 0 # Counter for keeping track of all profiles saved to file
        
        # Output Datasets -- For final stats and to write to file 
        tmp_file_names_ext = []
        tmp_file_names_sta = []
        
        # ----------------------------------------------------
        # ANALYSIS: Load in month by month and interpolate profile by profile
        for month_ii in range(0,n_months):
            
            print('Loading month of data: ' + str(current_month))
            
            # Make filenames for current month and read NEMO/EN4
            # If this files (e.g. file does not exist), skip to next month
            try:
                fn_nemo_month = make_nemo_filename(dn_nemo_data, current_month, nemo_file_suffix)
                fn_en4_month = make_en4_filename(dn_en4, current_month, en4_file_prefix)
                 
                print(fn_nemo_month)
                print(fn_en4_month)

                mod_month = read_monthly_model_nemo(fn_nemo_month, fn_nemo_domain)
                print('hi')
                obs_month = read_monthly_profile_en4(fn_en4_month)
            except:
                current_month = current_month + relativedelta(months=+1)
            
                print('       !!!Problem with read: Not analyzed ')
                continue
            
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
            print(obs_month)
            
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
            mod_profiles = mod_profiles.load()
            
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
            
            # Determine the current season integer
            current_season = month_season_dict[current_month.month]
            
            # Loop over profiles, interpolate model to obs depths and store in
            # monthly arrays
            ind_prof_use = []
            fail_reason = np.zeros(n_prof) # Debugging variable
            for prof in range(0,n_prof):
                
                # Select profile
                mod_profile = mod_profiles.isel(profile = prof)
                obs_profile = obs_month.isel(profile = prof)
                
                # If the nearest neighbour interpolation is bad, then skip the 
                # vertical interpolation -> keep profile as nans in monthly array
                if all(np.isnan(mod_profile.temperature)):
                    profiles_analysed += 1
                    fail_reason[prof] = 1
                    continue
                
                # Check that model point is within threshold distance of obs
                # If not, skip vertical interpolation -> keep profile as nans
                interp_dist = coastgu.calculate_haversine_distance(
                                                     obs_profile.longitude, 
                                                     obs_profile.latitude, 
                                                     mod_profile.longitude, 
                                                     mod_profile.latitude)
                if interp_dist > dist_crit:
                    profiles_analysed+=1
                    fail_reason[prof] = 2
                    continue
                
                # ----------------------------------------------------
                # Analysis 1) Interpolate model to obs depths using a linear interp
                obs_profile = obs_profile.rename({'z_dim':'depth'})
                obs_profile = obs_profile.set_coords('depth')
                
                # Check for NaN values in depth variables (breaks interpolation)
                #ind_notnan = ~xr.ufuncs.isnan(obs_profile.depth)
                #obs_profile = obs_profile.isel(depth=ind_notnan)
                
                mod_profile = mod_profile.rename({'z_dim':'depth_0'})
                
                # If interpolation fails for some reason, skip to next iteration
                try:
                    mod_profile_int = mod_profile.interp(depth_0 = obs_profile.depth)
                except:
                    profiles_analysed+=1
                    fail_reason[prof] = 3
                    continue
                
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
                profiles_analysed+=1
                
            print('       Interpolated Profiles.')
            # Find the union of masks for each variable
            mask_tem = np.logical_or(np.isnan(mod_tem), np.isnan(obs_tem))
            mask_sal = np.logical_or(np.isnan(mod_sal), np.isnan(obs_sal))
            mask_rho = np.logical_or(np.isnan(mod_rho), np.isnan(obs_rho))
            
            mod_tem[mask_tem] = np.nan
            obs_tem[mask_tem] = np.nan
            mod_sal[mask_sal] = np.nan
            obs_sal[mask_sal] = np.nan
            
            # Monthly stats xarray dataset - for file output and easy indexing
            season_save_vector = np.ones( n_prof, dtype=int )*current_season
            data = xr.Dataset(coords = dict(
                                      longitude=(["profile"], obs_month.longitude),
                                      latitude=(["profile"], obs_month.latitude),
                                      time=(["profile"], obs_month.time),
                                      level=(['level'], np.arange(0,400))),
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
                                      obs_z = (['profile','level'], obs_z)))
            
            analysis = xr.Dataset(coords = dict(
                                      longitude=(["profile"], obs_month.longitude),
                                      latitude=(["profile"], obs_month.latitude),
                                      time=(["profile"], obs_month.time),
                                      level=(['level'], np.arange(0,400))),
                              data_vars = dict(
                                      season = (['profile'], season_save_vector)))
            
            # Keep the profiles we want to keep
            n_prof_use = len(ind_prof_use)
            analysis = analysis.isel(profile = ind_prof_use)
            data = data.isel(profile = ind_prof_use)
            
            # ----------------------------------------------------
            # Analysis 3) All Anomalies
            # Errors at all depths
            analysis["error_tem"] = data.mod_tem - data.obs_tem
            analysis["error_sal"] = data.mod_sal - data.obs_sal
            
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
            
            print('       Basic errors done. ')
            
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
            
            print(mod_surf_tem)
            print(obs_surf_tem)
 
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
            
            print('       Surface and bottom errors done. ')
            
            # ----------------------------------------------------
            # Analysis 6) Mixed Layer Depth - needs GSW EOS
            # Absolute Pressure 
            # surface_ind = ref_depth<= mld_reference_depth
            # data_surf = data.isel(depth=surface_ind)
            # reference_s0_obs = np.nanmean(data_surf.obs_s0, axis=1)
            # reference_s0_mod = np.nanmean(data_surf.mod_s0, axis=1)
            # s0_diff_obs = np.abs( (data.obs_s0.T - reference_s0_obs).T )
            # s0_diff_mod = np.abs( (data.mod_s0.T - reference_s0_mod).T )
            # ind_mld_obs = (s0_diff_obs > mld_threshold).argmax(dim='depth')
            # ind_mld_mod = (s0_diff_mod > mld_threshold).argmax(dim='depth')
            # mld_obs = ref_depth[ind_mld_obs].astype(float)
            # mld_mod = ref_depth[ind_mld_mod].astype(float)
            # mld_obs[mld_obs<mld_reference_depth] = np.nan
            # mld_mod[mld_mod<mld_reference_depth] = np.nan
            
            # analysis['mld_obs'] = (['profile'], mld_obs)
            # analysis['mld_mod'] = (['profile'], mld_mod)
            
            # with open(fn_log, 'a') as f:
            #     f.write('       Mixed Layer Depth done. \n')
            
            save_mask_ind = np.zeros((n_regions, n_prof_use), dtype=bool)
            ind_season = [0, current_season] # 0 = whole year
                    
            # Loop over profiles and put errors into depth bins
            # < Currently a bit of a time bottleneck because of nested loops >
            for prof in range(0, data.dims['profile']):
                data_profile = data.isel(profile = prof)
                ana_profile = analysis.isel(profile=prof)
                
                # Find mean error by depth bins for current profile
                error_tem_bin = spst.binned_statistic(data_profile.obs_z, 
                                         ana_profile.error_tem, 'mean', 
                                         ref_depth_bins)[0]
                error_sal_bin = spst.binned_statistic(data_profile.obs_z, 
                                         ana_profile.error_sal, 'mean', 
                                         ref_depth_bins)[0]
                
                # Manual averaging if using 'sum' and not 'mean above
                # N_tem_tmp = spst.binned_statistic(data_profile.obs_z, 
                #                          ana_profile.error_tem, 'count', 
                #                          ref_depth_bins)[0]
                # N_sal_tmp = spst.binned_statistic(data_profile.obs_z, 
                #                          ana_profile.error_sal, 'count', 
                #                          ref_depth_bins)[0]
                
                # Loop over regional masks and determine if profile lies within
                for region_ii in range(0, n_regions):
                    # Get mask
                    mask = regional_masks[region_ii]
                    # If index is True then profile is in region
                    is_in_region = mask[ind2D[1][ind_prof_use][prof], 
                                      ind2D[0][ind_prof_use][prof]]
                    save_mask_ind[region_ii] = is_in_region
                    
                    if not is_in_region:
                        continue
                    
                    for ss in ind_season:
                        # Only add depths which don't contain NaNs
                        tem_notnan = ~np.isnan(error_tem_bin)
                        sal_notnan = ~np.isnan(error_sal_bin)
                        prof_error_tem[region_ii, ss, tem_notnan] += error_tem_bin[tem_notnan]
                        prof_error_sal[region_ii, ss, sal_notnan] += error_sal_bin[sal_notnan]
                        prof_error_tem_N[region_ii, ss, tem_notnan] += 1
                        prof_error_sal_N[region_ii, ss, sal_notnan] += 1
                        
                      
            # ----------------------------------------------------
            # WRITE 1) Write monthly stats to file
            
            # Postproc 1) Write data to file for each month - profiles stats
            yy = str(current_month.year)
            mm = str(current_month.month)
            
            # Create a diension coordinate that increases monotonically for
            # easy concatenation using xarray.open_mfdataset
            profile_id = np.arange(profiles_saved, profiles_saved + len(ind_prof_use) )
            profiles_saved = profiles_saved + len(ind_prof_use)
            analysis['profile'] = ('profile', profile_id)
            analysis['region_bool'] = (['region','profile'], save_mask_ind)
            
            # Create temp monthly file and write to it
            fn_tmp = 'en4_stats_by_profile_{0}{1}_{2}.nc'.format(yy,mm.zfill(2),run_name)
            fn = os.path.join(dn_output, fn_tmp)
            tmp_file_names_sta.append(fn)
            write_ds_to_file(analysis, fn, mode='w', unlimited_dims='profile')
            
            print('       File Written: ' + fn_tmp)
                
                
            # ----------------------------------------------------
            # WRITE 2) Write monthly extracted profiles to file
            
            # Create a diension coordinate that increases monotonically for
            # easy concatenation using xarray.open_mfdataset
            data['profile'] = ('profile', profile_id)
            data['region_bool'] = (['region','profile'], save_mask_ind)
            
            # Create temp monthly file and write to it
            fn_tmp = 'en4_extracted_profiles_{0}{1}_{2}.nc'.format(yy,mm.zfill(2), run_name)
            fn = os.path.join(dn_output, fn_tmp)
            tmp_file_names_ext.append(fn)
            write_ds_to_file(data, fn, mode='w', unlimited_dims='profile')
            
            print('       File Written: ' + fn_tmp)
            
            # Onto next month
            current_month = current_month + relativedelta(months=+1)
            
            ####
            ## END OF MONTH LOOP
            ####
            
        # Complete calculations of average profiles and regional statistics
        prof_error_tem = prof_error_tem/prof_error_tem_N
        prof_error_sal = prof_error_sal/prof_error_sal_N
         
        # Postproc 1) Write data to file for each month - profiles stats
        season_names = ['whole_year','DJF','MAM','JJA','SON']
        stats_regional = xr.Dataset( 
          data_vars = dict(
             prof_error_tem = (['region','season','depth'], prof_error_tem),
             prof_error_sal = (['region','season','depth'], prof_error_sal)),
          coords = dict(
             season_names = ('season', season_names),
             depth = ('depth', ref_depth)))
         
        fn_stats_region = 'en4_stats_regional_{0}.nc'.format(run_name)
        fn = os.path.join(dn_output, fn_stats_region)
        write_ds_to_file(stats_regional, fn, mode='w')
        
        # Concatenate monthly output files into one file
        fn_stats = 'en4_stats_profiles_{0}.nc'.format(run_name)
        fn = os.path.join(dn_output, fn_stats)
        all_stats = xr.open_mfdataset(tmp_file_names_sta, combine='by_coords', chunks={'profile':10000})
        write_ds_to_file(all_stats, fn)
        for ff in tmp_file_names_sta:
            os.remove(ff)
        
        fn_ext = 'en4_extracted_profiles_{0}.nc'.format(run_name)
        fn = os.path.join(dn_output, fn_ext)
        all_extracted = xr.open_mfdataset(tmp_file_names_ext,  combine='by_coords', chunks={'profile':10000})
        write_ds_to_file(all_extracted, fn)
        for ff in tmp_file_names_ext:
            os.remove(ff)

class regional_analysis():
    def __init__(self, fn_extracted, fn_stats, fn_nemo_data, fn_nemo_domain, 
                 fn_out, run_name, regional_masks, region_names):
        
        n_regions = len(regional_masks)
        
        ext = xr.open_dataset(fn_extracted, chunks={'profile':10000})
        nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain).dataset
        mask = nemo.bottom_level == 0
        
        ind2D = coastgu.nearest_indices_2D(nemo.longitude.values, nemo.latitude.values,
                                           ext.longitude.values, ext.latitude.values)
        
        is_in_region = [mm[ind2D[1], ind2D[0]] for mm in regional_masks]
        
        season_names = ['whole_year','DJF','MAM','JJA','SON']
        month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                             7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
        pd_time = pd.to_datetime(ext.time.values)
        pd_month = pd_time.month
        pd_season = np.array([month_season_dict[mm] for mm in pd_month])
        
        reg_list = [ext.isel(profile=np.where(rii)[0]) for rii in is_in_region]
        
        
        

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
                
                title_tmp = '$\Delta T$ | {0} | {1}'.format(region_names[rr], season_names[ss])
                fn_out = 'prof_error_tem_{0}_{1}{2}'.format(season_names[ss], region_abbrev[rr], file_type)
                fn_out = os.path.join(dn_out, fn_out)
                f,a = self.plot_profile_centred(tem_list[0].depth, tem_list,
                              title = title_tmp, legend_names = legend)
                print("  >>>>>  Saving: " + fn_out)
                f.savefig(fn_out)
                plt.close()
                
                title_tmp = '$\Delta S$ |' + region_names[rr] +' | '+season_names[ss]
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
            ax.plot(vv.squeeze(), depth.squeeze())
            
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