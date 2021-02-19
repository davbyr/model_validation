
"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.0 (20-01-2021)

"""

import coast
import coast.general_utils as coastgu
import coast.plot_util as plot_util
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
fn_profile_stats = "/Users/dbyrne/Projects/CO9_AMM15/data/en4_stats_by_profile*" 
fn_regional_stats = "/Users/dbyrne/Projects/CO9_AMM15/data/en4_stats_regional.nc"
dn_output_figs = "/Users/dbyrne/Projects/CO9_AMM15/data/figs/"

fn_test = "/Users/dbyrne/Projects/CO9_AMM15/data/en4/EN.4.2.1.f.profiles.g10.200403.nc"

run_name = 'CO9_AMM15_p0'


 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 #SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


def main():

    stats_profile = read_stats_profiles(fn_profile_stats)
    
    stats_regional = read_stats_regional(fn_regional_stats)
    
    ######################
    # SURFACE ERRORS
    ######################
    
    sf = 2
    
    # All surface errors - TEMPERATURE
    title_tmp = 'SST Anom. | Whole Year | ' + run_name
    fn_out = 'surf_error_tem_annual.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    geo_scatter_profiles(stats_profile.surf_error_tem, title_tmp, fn_out, sf=sf)
    
    # Winter surface errors TEMPERATURE
    title_tmp = 'SST Anom. | DJF |' + run_name
    fn_out = 'surf_error_tem_djf.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==1
    geo_scatter_profiles(stats_profile.surf_error_tem.isel(profile=ind_season),
                         title_tmp, fn_out, sf=sf)
    
    # Spring surface errors - TEMPERATURE
    title_tmp = 'SST Anom.| MAM |' + run_name
    fn_out = 'surf_error_tem_mam.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==2
    geo_scatter_profiles(stats_profile.surf_error_tem.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=sf)
    
    # Summer surface errors - TEMPERATURE
    title_tmp = 'SST Anom. | JJA |' + run_name
    fn_out = 'surf_error_tem_jja.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==3
    geo_scatter_profiles(stats_profile.surf_error_tem.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=sf)
    
    # Autumn surface errors - TEMPERATURE
    title_tmp = 'SST Anom. | SON |' + run_name
    fn_out = 'surf_error_tem_son.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==4
    geo_scatter_profiles(stats_profile.surf_error_tem.isel(profile=ind_season), 
                         title_tmp, fn_out)
    
    # All surface errors - SALINITY
    title_tmp = 'SSS Anom. | Whole Year | ' + run_name
    fn_out = 'surf_error_sal_annual.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    geo_scatter_profiles(stats_profile.surf_error_sal, title_tmp, fn_out, sf=2)
    
    # Winter surface errors- SALINITY
    title_tmp = 'SSS Anom. | DJF |' + run_name
    fn_out = 'surf_error_sal_djf.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==1
    geo_scatter_profiles(stats_profile.surf_error_sal.isel(profile=ind_season),
                         title_tmp, fn_out, sf=2)
    
    # Spring surface errors- SALINITY
    title_tmp = 'SSS Anom. | MAM |' + run_name
    fn_out = 'surf_error_sal_mam.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==2
    geo_scatter_profiles(stats_profile.surf_error_sal.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=2)
    
    # Summer surface errors- SALINITY
    title_tmp = 'SSS Anom.) | JJA |' + run_name
    fn_out = 'surf_error_sal_jja.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==3
    geo_scatter_profiles(stats_profile.surf_error_sal.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=2)
    
    # Autumn surface errors- SALINITY
    title_tmp = 'SSS Anom. | SON |' + run_name
    fn_out = 'surf_error_sal_son.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==4
    geo_scatter_profiles(stats_profile.surf_error_sal.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=2)
    
    ######################
    # BOTTOM ERRORS
    ######################
    
    # All bottace errors
    title_tmp = 'SBT Anom. | Whole Year | ' + run_name
    fn_out = 'bott_error_tem_annual.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    geo_scatter_profiles(stats_profile.bott_error_tem, title_tmp, fn_out, sf=sf)
    
    # Winter Bottom errors
    title_tmp = 'SBT Anom. | DJF |' + run_name
    fn_out = 'bott_error_tem_djf.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==1
    geo_scatter_profiles(stats_profile.bott_error_tem.isel(profile=ind_season),
                         title_tmp, fn_out, sf=sf)
    
    # Spring Bottom errors
    title_tmp = 'SBT Anom. | MAM |' + run_name
    fn_out = 'bott_error_tem_mam.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==2
    geo_scatter_profiles(stats_profile.bott_error_tem.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=sf)
    
    # Summer Bottom errors
    title_tmp = 'SBT Anom. | JJA |' + run_name
    fn_out = 'bott_error_tem_jja.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==3
    geo_scatter_profiles(stats_profile.bott_error_tem.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=sf)
    
    # Autumn Bottom errors
    title_tmp = 'SBT Anom. | SON |' + run_name
    fn_out = 'bott_error_tem_son.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==4
    geo_scatter_profiles(stats_profile.bott_error_tem.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=sf)
    
    # All bottace errors - SALINITY
    title_tmp = 'SBS Anom. | Whole Year | ' + run_name
    fn_out = 'bott_error_sal_annual.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    geo_scatter_profiles(stats_profile.bott_error_sal, title_tmp, fn_out, sf=2)
    
    # Winter Bottom errors - SALINITY
    title_tmp = 'SBS Anom. | DJF |' + run_name
    fn_out = 'bott_error_sal_djf.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==1
    geo_scatter_profiles(stats_profile.bott_error_sal.isel(profile=ind_season),
                         title_tmp, fn_out, sf=2)
    
    # Spring Bottom errors - SALINITY
    title_tmp = 'SBS Anom. | MAM |' + run_name
    fn_out = 'bott_error_sal_mam.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==2
    geo_scatter_profiles(stats_profile.bott_error_sal.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=2)
    
    # Summer Bottom errors - SALINITY
    title_tmp = 'SBS Anom. | JJA |' + run_name
    fn_out = 'bott_error_sal_jja.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==3
    geo_scatter_profiles(stats_profile.bott_error_sal.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=2)
    
    # Autumn Bottom errors - SALINITY
    title_tmp = 'SBS Anom. | SON |' + run_name
    fn_out = 'bott_error_sal_son.eps'
    fn_out = os.path.join(dn_output_figs, fn_out)
    ind_season = stats_profile.season==4
    geo_scatter_profiles(stats_profile.bott_error_sal.isel(profile=ind_season), 
                         title_tmp, fn_out, sf=2)
    
    ######################
    # SURFACE CRPS
    ######################
    try:
        # All surface errors - TEMPERATURE
        title_tmp = 'Surface Temperature CRPS (5m Mean) | Whole Year | ' + run_name
        fn_out = 'surf_crps_tem_annual.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        geo_scatter_profiles_abs(stats_profile.surf_crps_tem, title_tmp, fn_out)
        
        # Winter surface CRPSs TEMPERATURE
        title_tmp = 'Surface Temperature CRPS (5m Mean) | DJF |' + run_name
        fn_out = 'surf_crps_tem_djf.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        ind_season = stats_profile.season==1
        geo_scatter_profiles_abs(stats_profile.surf_crps_tem.isel(profile=ind_season),
                             title_tmp, fn_out)
        
        # Spring surface CRPSs - TEMPERATURE
        title_tmp = 'Surface Temperature CRPS (5m Mean) | MAM |' + run_name
        fn_out = 'surf_crps_tem_mam.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        ind_season = stats_profile.season==2
        geo_scatter_profiles_abs(stats_profile.surf_crps_tem.isel(profile=ind_season), 
                             title_tmp, fn_out)
        
        # Summer surface CRPSs - TEMPERATURE
        title_tmp = 'Surface Temperature CRPS (5m Mean) | JJA |' + run_name
        fn_out = 'surf_crps_tem_jja.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        ind_season = stats_profile.season==3
        geo_scatter_profiles_abs(stats_profile.surf_crps_tem.isel(profile=ind_season), 
                             title_tmp, fn_out)
        
        # Autumn surface CRPSs - TEMPERATURE
        title_tmp = 'Surface Temperature CRPS (5m Mean) | SON |' + run_name
        fn_out = 'surf_crps_tem_son.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        ind_season = stats_profile.season==4
        geo_scatter_profiles_abs(stats_profile.surf_crps_tem.isel(profile=ind_season), 
                             title_tmp, fn_out)
        
        # All surface CRPSs - SALINITY
        title_tmp = 'Surface Salinity CRPS (5m Mean) | Whole Year | ' + run_name
        fn_out = 'surf_crps_sal_annual.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        geo_scatter_profiles_abs(stats_profile.surf_crps_sal, title_tmp, fn_out, sf=2)
        
        # Winter surface CRPSs- SALINITY
        title_tmp = 'Surface Salinity CRPS (5m Mean) | DJF |' + run_name
        fn_out = 'surf_crps_sal_djf.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        ind_season = stats_profile.season==1
        geo_scatter_profiles_abs(stats_profile.surf_crps_sal.isel(profile=ind_season),
                             title_tmp, fn_out, sf=2)
        
        # Spring surface CRPSs- SALINITY
        title_tmp = 'Surface Salinity CRPS (5m Mean) | MAM |' + run_name
        fn_out = 'surf_crps_sal_mam.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        ind_season = stats_profile.season==2
        geo_scatter_profiles_abs(stats_profile.surf_crps_sal.isel(profile=ind_season), 
                             title_tmp, fn_out, sf=2)
        
        # Summer surface CRPSs- SALINITY
        title_tmp = 'Surface Salinity CRPS (5m Mean) | JJA |' + run_name
        fn_out = 'surf_crps_sal_jja.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        ind_season = stats_profile.season==3
        geo_scatter_profiles_abs(stats_profile.surf_crps_sal.isel(profile=ind_season), 
                             title_tmp, fn_out, sf=2)
        
        # Autumn surface CRPSs- SALINITY
        title_tmp = 'Surface Salinity CRPS (5m Mean) | SON |' + run_name
        fn_out = 'surf_crps_sal_son.eps'
        fn_out = os.path.join(dn_output_figs, fn_out)
        ind_season = stats_profile.season==4
        geo_scatter_profiles_abs(stats_profile.surf_crps_sal.isel(profile=ind_season), 
                             title_tmp, fn_out, sf=2)
    except:
        pass
    ######################
    # MEAN PROFILES
    ######################
    
    # For titles
    region_names = ['Whole Domain','<200m','>200m','North Sea','Outer Shelf',
                    'Norwegian Trench','English Channel']
    # For file names
    region_abbrev = ['wholedomain','shallow','deep','northsea',
                     'outershelf','nortrench','engchannel']
    season_names = ['Annual','DJF','MAM','JJA','SON']
    n_regions = len(region_names)
    n_seasons = len(season_names)
    for rr in range(0,n_regions):
        for ss in range(0,n_seasons):
            stats_regional_tmp = stats_regional.isel(region=rr, season=ss,
                                                     depth=np.arange(0,150))
            
            title_tmp = 'Mean Temperature Profiles (degC) | '+region_names[rr]+' | '+season_names[ss]
            fn_out = 'mean_prof_tem_'+season_names[ss]+'_'+region_abbrev[rr]+'.eps'
            fn_out = os.path.join(dn_output_figs, fn_out)
            legend = ['Model','EN4']
            plot_profile(stats_regional_tmp.depth[4:], 
                          [stats_regional_tmp.mean_prof_mod_tem[4:], 
                          stats_regional_tmp.mean_prof_obs_tem[4:]],
                          title = title_tmp, fn_out = fn_out,
                          legend_names = legend)
            
            title_tmp = 'Mean Salinity Profiles (PSU) | '+region_names[rr]+' | '+season_names[ss]
            fn_out = 'mean_prof_sal_'+season_names[ss]+'_'+region_abbrev[rr]+'.eps'
            fn_out = os.path.join(dn_output_figs, fn_out)
            legend = ['Model','EN4']
            plot_profile(stats_regional_tmp.depth[4:], 
                          [stats_regional_tmp.mean_prof_mod_sal[4:], 
                          stats_regional_tmp.mean_prof_obs_sal[4:]],
                          title = title_tmp, fn_out = fn_out, 
                          legend_names = legend)
            
            legend = ['CO9_AMM15p0']
            
            title_tmp = '$\Delta T$ |' + region_names[rr] +' | '+season_names[ss]
            fn_out = 'prof_error_tem_'+season_names[ss]+'_'+region_abbrev[rr]+'.eps'
            fn_out = os.path.join(dn_output_figs, fn_out)
            plot_profile_centred(stats_regional_tmp.depth[4:], 
                         stats_regional_tmp.prof_error_tem[4:],
                         title = title_tmp, fn_out = fn_out,
                         legend_names = legend)
            
            title_tmp = '$\Delta S$ |' + region_names[rr] +' | '+season_names[ss]
            fn_out = 'prof_error_sal_'+season_names[ss]+'_'+region_abbrev[rr]+'.eps'
            fn_out = os.path.join(dn_output_figs, fn_out)
            plot_profile_centred(stats_regional_tmp.depth[4:], 
                         stats_regional_tmp.prof_error_sal[4:],
                         title = title_tmp, fn_out = fn_out,
                         legend_names = legend)
        
def plot_profile(depth, variables, title, fn_out, legend_names= {} ):

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

def plot_profile_centred(depth, variables, title, fn_out, legend_names= {} ):

    fig = plt.figure(figsize=(3.5,7))
    ax = plt.subplot(111)
    
    if type(variables) is not list:
        variables = [variables]

    for vv in variables:
        ax.plot(vv.squeeze(), depth.squeeze())
        
    xmax = np.nanmax(np.abs(vv))
    plt.xlim(-xmax-0.05*xmax, xmax+0.05*xmax)
    ymax = np.nanmax(np.abs(depth))
    plt.plot([0,0],[-1e7,1e7], linestyle='--',linewidth=1,color='k')
    plt.ylim(0,ymax)
    plt.gca().invert_yaxis()
    plt.ylabel('Depth (m)')
    plt.grid()
    plt.legend(legend_names)
    plt.title(title, fontsize=12)
    print("  >>>>>  Saving: " + fn_out)
    plt.savefig(fn_out)
    plt.close()
    return fig, ax

def geo_scatter_profiles(var, title, fn_out, sf=3):
    cmax = np.nanmean(var) + sf*np.nanstd(var)
    cmin = np.nanmean(var) - sf*np.nanstd(var)
    scatter_kwargs = {"cmap":"seismic", "vmin":-2, "vmax":2, 'marker':'s',
                      'edgecolors':None}
    sca = plot_util.geo_scatter(var.longitude, var.latitude, 
                c = var, s=.5, scatter_kwargs=scatter_kwargs, title = title)
    print("  >>>>>  Saving: " + fn_out)
    plt.savefig(fn_out)
    plt.close()
    return

def geo_scatter_profiles_abs(var, title, fn_out, sf=3):
    cmax = np.nanmean(var) + sf*np.nanstd(var)
    cmin = np.nanmean(var) - sf*np.nanstd(var)
    cmin = np.max([cmin, 0])
    scatter_kwargs = {"cmap":"viridis", "vmin":cmin, "vmax":cmax}
    sca = plot_util.geo_scatter(var.longitude, var.latitude, 
                c = var, s=1, scatter_kwargs=scatter_kwargs, title = title)
    print("  >>>>>  Saving: " + fn_out)
    plt.savefig(fn_out)
    plt.close()
    return


def read_stats_profiles(fn_profile_stats):
    return xr.open_mfdataset(fn_profile_stats, chunks={})

def read_stats_regional(fn_regional_stats):
    return xr.open_dataset(fn_regional_stats, chunks={})

if __name__ == '__main__':
    main()