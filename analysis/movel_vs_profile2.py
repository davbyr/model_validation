"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.0 (20-01-2021)

A script for validation of monthly mean model temperature and salinity against
EN4 profile data. 

Parts of this script are modular and are designed to be switched out where
necessary. For example, differently structured input data files. The beginning
of the script has two routines for reading and structuring data:
    
    read_monthly_model_nemo() and
    read_profile_monthly_en4()
    
Both of these routines should read model and EN4 profile data (respectively)
into xarray object that adhere to COAsT data structure guidelines. So for
NEMO data:
    
    
and EN4 data:
    
Each should contan "temperature" and "salinity" variables. This is what will
be compared. The input NEMO data should be monthly mean data, with all depths.
The NEMO data should have no gaps. If there are gaps, then multiple analyses
should be run. There are example reading scripts included below. These may or
may not work for you. If you change them, as long as the correct data format
is adhered to, the rest of the script should continue to work.

The script will then:
    
    1. Preprocessing
        a. Cut down the EN4 obs to just those over the model domain.
        b. Cit down the EN4 obs to just the model time window.
        c. Initialise output arrays.
        
    2. Pre-Analysis
        a. Loads and analyses data in monthly chunks.
        b. Identifies nearest model grid cells to profile data.
           Uses this to extract an equivalent model profile.
        c. Interpolates both obs and model profiles to a set of reference
           depths.
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
        a. Mean profiles for each region and season
        b. Mean profile statistics (from the Profile Analysis) for each region
           and season.
    
    5. Postprocessing
        a. Profile analysis written to file monthly.
        b. Regional analysis written to file at the end of script.
        

        

"""
import sys
sys.path.append('/Users/dbyrne/code/COAsT/')

import coast
import coast.general_utils as coastgu
import numpy as np
import datetime as datetime
import pandas as pd
import gsw
import xarray as xr
import sys
import os.path
import glob
from dateutil.relativedelta import *
import scipy.stats as spst

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # GLOBAL VARIABLES
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
# Input paths and Filenames
dn_nemo_data = '/Users/dbyrne/Projects/CO9_AMM15/data/nemo/'
fn_nemo_domain = '/Users/dbyrne/Projects/CO9_AMM15/data/nemo/CO7_EXACT_CFG_FILE.nc'
dn_en4 = '/Users/dbyrne/Projects/CO9_AMM15/data/en4/'

# Output files
dn_output_data = "/Users/dbyrne/Projects/CO9_AMM15/data/analysis/p0"
fn_output_region_data = "/Users/dbyrne/Projects/CO9_AMM15/data/analysis/p0/stats_region.nc"

# Definitions
surface_thickness = 5 # Definition of surface: distance from top to average
bottom_thickness = 10 # Definition of bottom: distance from the bottom (% of depth)

mld_reference_depth = 5
mld_threshold = 0.02

start_month = datetime.datetime(2004,2,1) 
end_month = datetime.datetime(2014,12,1) 

# Diagnostics & Stats
do_crps =  False
save_extract_profiles = True

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 #SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


def main():
    
    current_month = start_month
    n_months = (end_month.year - start_month.year)*12 + \
                    (end_month.month - start_month.month) + 1
    fn_nemo_month = make_nemo_filename(dn_nemo_data, current_month)
    mod_month = mod_month = read_monthly_model_nemo(fn_nemo_month, fn_nemo_domain)
    
    fn_log = os.path.join(dn_output_data, 'log_profile.txt')
    with open(fn_log, 'w') as f:
        f.write(' *Profile analysis starting.* \n')
    
    n_obs_levels = 400
    
    regional_masks = define_regional_masks(mod_month.bathymetry) # Define list of regional masks
    n_regions = len(regional_masks) # Number of regions
    
    month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                         7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
    season_names = ['whole_year','DJF','MAM','JJA','SON']
    n_seasons = len(season_names) # 0 = all year, 1-4 = winter -> autumn
    
    # DEPTH BIN EDGES
    ref_depth_bins = np.concatenate([ np.arange(-0.001,200,5) ])
    bin_widths = ref_depth_bins[1:] - ref_depth_bins[:-1]
    ref_depth = ref_depth_bins[:-1] + .5*bin_widths
    n_ref_depth_bins = len(ref_depth_bins)
    n_ref_depth = len(ref_depth)
    
    prof_error_tem = np.zeros((n_regions, n_seasons, n_ref_depth))
    prof_error_sal = np.zeros((n_regions, n_seasons, n_ref_depth))
    prof_error_tem_N = np.zeros((n_regions, n_seasons, n_ref_depth))
    prof_error_sal_N = np.zeros((n_regions, n_seasons, n_ref_depth))
    
    profiles_analysed = 0 # Counter for keeping track of all profiles analyzed
    profiles_saved = 0 # Counter for keeping track of all profiles saved to file
    
    # Output Datasets -- For final stats and to write to file 
    tmp_file_names = []
    
    # ----------------------------------------------------
    # ANALYSIS: Load in month by month and analyze profile by profile
    for month_ii in range(0,n_months):
        
        # Write to screen
        with open(fn_log, 'a') as f:
            f.write('Loading month of data: ' + str(current_month) + '\n')
        
        # Extract month from model and observations
        try:
            fn_nemo_month = make_nemo_filename(dn_nemo_data, current_month)
            fn_en4_month = make_en4_filename(dn_en4, current_month)
            
            mod_month = read_monthly_model_nemo(fn_nemo_month, fn_nemo_domain)
            obs_month = read_monthly_profile_en4(fn_en4_month)
        except:
            current_month = current_month + relativedelta(months=+1)
        
            with open(fn_log, 'a') as f:
                f.write('       !!!Problem with read: Not analyzed' + fn_tmp + ' \n')
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
        obs_month = obs_month.isel(profile=ind) # FOR TESTING
        obs_month = obs_month.load()
        
        # ----------------------------------------------------
        # Init 4) Get model indices (space and time) corresponding to observations
        ind2D = coastgu.nearest_indices_2D(mod_month['longitude'], 
                                           mod_month['latitude'],
                                           obs_month['longitude'], 
                                           obs_month['latitude'], 
                                           mask=mod_month.landmask)
        
        # Loop over profiles in this month
        n_prof = len(obs_month.time) # Number of profiles in this month
        mod_profiles = mod_month.isel(x_dim=ind2D[0], y_dim=ind2D[1])
        mod_profiles = mod_profiles.rename({'dim_0':'profile'})
        mod_profiles = mod_profiles.load()
        
        # Define monthly arrays
        mod_tem = np.zeros((n_prof , n_obs_levels))*np.nan
        obs_tem = np.zeros((n_prof, n_obs_levels))*np.nan
        mod_sal = np.zeros((n_prof, n_obs_levels))*np.nan
        obs_sal = np.zeros((n_prof, n_obs_levels))*np.nan
        mod_rho = np.zeros((n_prof, n_obs_levels))*np.nan
        obs_rho = np.zeros((n_prof, n_obs_levels))*np.nan
        mod_s0 = np.zeros((n_prof, n_obs_levels))*np.nan
        obs_s0 = np.zeros((n_prof, n_obs_levels))*np.nan
        obs_z = np.zeros((n_prof, n_obs_levels))*np.nan
        
        # Determine the current season
        current_season = month_season_dict[current_month.month]
        
        # Loop over profiles, interpolate to common depths and store in
        # monthly arrays
        ind_prof_use = []
        fail_reason = np.zeros(n_prof)*np.nan # Debugging variable
        for prof in range(0,n_prof):
            
            mod_profile = mod_profiles.isel(profile = prof)
            obs_profile = obs_month.isel(profile = prof)
            
            # If the nearest neighbour interpolation is bad, then skip the 
            # vertical interpolation -> keep profile as nans in monthly array
            if all(np.isnan(mod_profile.temperature)):
                profiles_analysed += 1
                fail_reason[prof] = 0
                continue
            
            # Check that model point is within threshold distance of obs
            # If not, skip vertical interpolation -> keep profile as nans
            dist_crit = 10 # Interpolation distance at which to discard
            interp_dist = coastgu.calculate_haversine_distance(
                                                 obs_profile.longitude, 
                                                 obs_profile.latitude, 
                                                 mod_profile.longitude, 
                                                 mod_profile.latitude)
            if interp_dist > dist_crit:
                profiles_analysed+=1
                fail_reason[prof] = 1
                continue
            
            # ----------------------------------------------------
            # Analysis 1) Interpolate model and obs to reference depths
            obs_profile = obs_profile.rename({'z_dim':'depth'})
            obs_profile = obs_profile.set_coords('depth')
            
            # Check for NaN values in depth variables (breaks interpolation)
            #ind_notnan = ~xr.ufuncs.isnan(obs_profile.depth)
            #obs_profile = obs_profile.isel(depth=ind_notnan)
            
            mod_profile = mod_profile.rename({'z_dim':'depth_0'})
            
            # If interpolation fails for some reason, skip to next iteration
            try:
                #obs_profile_1m = obs_profile.interp(depth = ref_depth)
                mod_profile_int = mod_profile.interp(depth_0 = obs_profile.depth)
            except:
                profiles_analysed+=1
                fail_reason[prof] = 2
                continue
            
            # ----------------------------------------------------
            # Analysis 2) Calculate Density per Profile
            
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
            
        with open(fn_log, 'a') as f:
            f.write('       Interpolated Profiles. \n')
        
        # Find the union of masks for each variable
        mask_tem = np.logical_or(np.isnan(mod_tem), np.isnan(obs_tem))
        mask_sal = np.logical_or(np.isnan(mod_sal), np.isnan(obs_sal))
        mask_rho = np.logical_or(np.isnan(mod_rho), np.isnan(obs_rho))
        
        mod_tem[mask_tem] = np.nan
        obs_tem[mask_tem] = np.nan
        mod_sal[mask_sal] = np.nan
        obs_sal[mask_sal] = np.nan
        
        # Monthly stats xarray dataset
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
        
        with open(fn_log, 'a') as f:
            f.write('       Basic errors done. \n')
        
        # ----------------------------------------------------
        # Analysis 4) Surface stats
        # Get indices corresponding to surface depth
        # Get averages over surface depths
        analysis['surface_definition'] = surface_thickness
        surface_ind = data.obs_z <= surface_thickness
        
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
        analysis['bottom_definition'] = bottom_thickness
        prof_bathy = mod_profiles.bathymetry.isel(profile=ind_prof_use)
        percent_depth = bottom_thickness
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
        
        with open(fn_log, 'a') as f:
            f.write('       Surface and bottom errors done. \n')
        
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
        
        with open(fn_log, 'a') as f:
            f.write('       Mixed Layer Depth done. \n')
        
        # ----------------------------------------------------
        # Analysis 6) Surface CRPS 
        if do_crps:
            crps_tem = np.zeros(n_prof_use)*np.nan
            crps_sal = np.zeros(n_prof_use)*np.nan
            nh_ind = coastgu.subset_indices_by_distance_BT(mod_month.longitude,
                                                        mod_month.latitude,
                                                        data.longitude, 
                                                        data.latitude,
                                                        radius = 5)
            for nh_ii in range(0,n_prof_use):
                # Couple of checks on neighbourhood outcome
                if np.size(nh_ind[1][nh_ii]) <=1:
                    continue
                if len(nh_ind[1][nh_ii]) < 1:
                    continue
                
                mod_nh = mod_month.isel(x_dim = xr.DataArray(nh_ind[1][nh_ii]), 
                                        y_dim = xr.DataArray(nh_ind[0][nh_ii]))
        
                surf_ind = mod_nh.depth_0 <= surface_thickness
                tem_nh = np.array(mod_nh.temperature)
                sal_nh = np.array(mod_nh.salinity)
                tem_nh[~surf_ind] = np.nan
                sal_nh[~surf_ind] = np.nan
                surf_tem_crps = np.nanmean(tem_nh, axis=0)
                surf_sal_crps = np.nanmean(sal_nh, axis=0)
                
                crps_tem[nh_ii] = coast.crps_util.crps_empirical(surf_tem_crps,
                                                          obs_surf_tem[nh_ii])
                crps_sal[nh_ii] = coast.crps_util.crps_empirical(surf_sal_crps,
                                                          obs_surf_sal[nh_ii])
                analysis['surf_crps_tem'] = ('profile', crps_tem)
                analysis['surf_crps_sal'] = ('profile', crps_sal)
            
        with open(fn_log, 'a') as f:
            f.write('       Surface CRPS done. \n')
            
        # ----------------------------------------------------
        # Analysis 8) Regional stats - averages over defined regions and seasons
        
        save_mask_ind = np.zeros((n_regions, n_prof_use), dtype=bool)
        ind_season = [0, current_season] # 0 = whole year
                
        for prof in range(0, data.dims['profile']):
            data_profile = data.isel(profile = prof)
            ana_profile = analysis.isel(profile=prof)
            
            error_tem_bin = spst.binned_statistic(data_profile.obs_z, 
                                     ana_profile.error_tem, 'mean', 
                                     ref_depth_bins)[0]
            error_sal_bin = spst.binned_statistic(data_profile.obs_z, 
                                     ana_profile.error_sal, 'mean', 
                                     ref_depth_bins)[0]
            
            # N_tem_tmp = spst.binned_statistic(data_profile.obs_z, 
            #                          ana_profile.error_tem, 'count', 
            #                          ref_depth_bins)[0]
            # N_sal_tmp = spst.binned_statistic(data_profile.obs_z, 
            #                          ana_profile.error_sal, 'count', 
            #                          ref_depth_bins)[0]
            
            for region_ii in range(0, n_regions):
                mask = regional_masks[region_ii]
                is_in_region = mask[ind2D[1][ind_prof_use][prof], 
                                  ind2D[0][ind_prof_use][prof]]
                save_mask_ind[region_ii] = is_in_region
                
                if not is_in_region:
                    continue
                
                for ss in ind_season:
                    # Profile errors
                    tem_notnan = ~np.isnan(error_tem_bin)
                    sal_notnan = ~np.isnan(error_sal_bin)
                    prof_error_tem[region_ii, ss, tem_notnan] += error_tem_bin[tem_notnan]
                    prof_error_sal[region_ii, ss, sal_notnan] += error_tem_bin[sal_notnan]
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
        fn_tmp = 'en4_stats_by_profile_'+yy+mm.zfill(2)+'.nc'
        tmp_file_names.append(fn_tmp)
        fn = os.path.join(dn_output_data, fn_tmp)
        analysis.to_netcdf(fn, mode='w', unlimited_dims='profile')
        
        with open(fn_log, 'a') as f:
            f.write('       File Written: ' + fn_tmp + ' \n')
            
            
        # ----------------------------------------------------
        # WRITE 2) Write monthly extracted profiles to file
        
        # Create a diension coordinate that increases monotonically for
        # easy concatenation using xarray.open_mfdataset
        data['profile'] = ('profile', profile_id)
        data['region_bool'] = (['region','profile'], save_mask_ind)
        
        # Create temp monthly file and write to it
        fn_tmp = 'en4_extracted_profiles_'+yy+mm.zfill(2)+'.nc'
        tmp_file_names.append(fn_tmp)
        fn = os.path.join(dn_output_data, fn_tmp)
        data.to_netcdf(fn, mode='w', unlimited_dims='profile')
        
        with open(fn_log, 'a') as f:
            f.write('       File Written: ' + fn_tmp + ' \n')
        
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
     
    fn_stats_region = 'en4_stats_regional.nc'
    fn = os.path.join(dn_output_data, fn_stats_region)
    stats_regional.to_netcdf(fn, mode='w')

def read_monthly_profile_en4(fn_en4):
    '''
    '''
    
    en4 = coast.PROFILE()
    en4.read_EN4(fn_en4, chunks={})
    en4.dataset = en4.dataset.rename({'practical_salinity':'salinity'})
    
    return en4.dataset[['temperature','salinity','depth']]

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

def define_regional_masks(bath):
    '''
    Returns a list of regional masks
    '''
    n_r, n_c = bath.shape
    whole_domain = region_def_whole_domain(n_r, n_c)
    north_sea = region_def_north_sea(bath)
    outer_shelf = region_def_outer_shelf(bath)
    norwegian_trench = region_def_norwegian_trench(bath)
    english_channel = region_def_english_channel(bath)
    shallow = region_def_shallow(bath)
    deep = region_def_deep(bath)
    regional_masks = [whole_domain, shallow, deep, 
                      north_sea, outer_shelf, norwegian_trench, english_channel]
    return regional_masks

def region_def_whole_domain(n_r, n_c):
    return np.ones((n_r,n_c))

def region_def_shallow(bath):
    return bath<=200

def region_def_deep(bath):
    return bath>200

def region_def_north_sea(bath):
    return ( (bath.latitude<60.5) & 
           (bath.latitude>54.09) &
           (bath<200) &
           (bath.longitude < 9) &
           (bath.latitude - 0.83*bath.longitude < (58.641 + 0.83*3.316) ) & 
           (bath.latitude +0.606*bath.longitude > (53.808 -0.606*0.227) ) &
           ~( (bath.longitude > 4.126) & (bath.latitude>58.59)  ) &
           ~( (bath.longitude > 5) & (bath.latitude>58.121)  ) &
           ~( (bath.longitude >6.3) & (bath.latitude>57.859)  ) & 
           ~( (bath.longitude >7.5) & (bath.latitude<55.5)  ))

def region_def_outer_shelf(bath):
    return ( ( (bath.latitude + 0.704*bath.longitude < (48.66 - 0.704*4.532) ) 
               | (bath.latitude - 0.83*bath.longitude > (58.641 + 0.83*3.316) )  
               | ( (bath.latitude > 60.5) & (bath.longitude > -1.135) & (bath.longitude < 3.171)) ) & 
             ~( (bath.longitude < -3.758) & (bath.latitude > 60.448) ) &
             ~( (bath.longitude < -11.468) & (bath.latitude > 55.278) ) &
             (bath<200) &
             (bath.latitude > 48) )

def region_def_norwegian_trench(bath):
    return ( (bath >200) &
             (bath.longitude > 1.12) & 
             (bath.longitude < 10.65) &
             (bath.latitude < 61.83 ) )

def region_def_english_channel(bath):
    return ( (~region_def_outer_shelf(bath)) &
            (~region_def_north_sea(bath)) &
            (bath.latitude < 55) &
            (bath.latitude > 48) &
            (bath.longitude > -4.8) &
            ~( (bath.latitude > 50.5) & (bath.longitude <-2) ) &
            (bath < 200) & 
            (bath > 0))

def make_nemo_filename(dn, date):
    suffix = '01_monthly_grid_T'
    month = str(date.month).zfill(2)
    year = date.year
    yearmonth = str(year) + str(month)
    return os.path.join(dn, yearmonth + suffix + '.nc')

def make_en4_filename(dn, date):
    prefix = 'EN.4.2.1.f.profiles.g10.'
    month = str(date.month).zfill(2)
    year = date.year
    yearmonth = str(year) + str(month)
    return os.path.join(dn, prefix + yearmonth + '.nc')

if __name__ == '__main__':
    main()