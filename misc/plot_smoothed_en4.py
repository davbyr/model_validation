import sys
sys.path.append('/Users/dbyrne/code/COAsT/')
import matplotlib.pyplot as plt
import coast
import coast.general_utils as gu
import numpy as np
import coast.plot_util as plot_util
import xarray as xr
import os.path

radius = 25
run_name = 'CO9_AMM15p0'
seasons = ['Annual','DJF','MAM','JJA','SON']
n_seasons = len(seasons)
fn_smoothed = "/Users/dbyrne/Projects/CO9_AMM15/data/analysis/co7/en4_profile_stats_smoothed.nc"
dn_output_figs = "/Users/dbyrne/Projects/CO9_AMM15/data/figs/co7"

smoothed = xr.open_dataset(fn_smoothed)
lon = smoothed.longitude
lat = smoothed.latitude
file_type = '.png'

N_crit =5
    
for season in range(1, len(seasons)):
    
    tmp = smoothed.isel(season=season)
    
    surf_error_tem = tmp.surf_error_tem.values
    surf_error_tem = tmp.surf_error_tem.values
    surf_tem_N = tmp.surf_tem_N.values
    surf_tem_N = tmp.surf_tem_N.values
    bott_error_tem = tmp.bott_error_tem.values
    bott_error_tem = tmp.bott_error_tem.values
    bott_tem_N = tmp.bott_tem_N.values
    bott_tem_N = tmp.bott_tem_N.values
    
    surf_error_sal = tmp.surf_error_sal.values
    surf_error_sal = tmp.surf_error_sal.values
    surf_sal_N = tmp.surf_sal_N.values
    surf_sal_N = tmp.surf_sal_N.values
    bott_error_sal = tmp.bott_error_sal.values
    bott_error_sal = tmp.bott_error_sal.values
    bott_sal_N = tmp.bott_sal_N.values
    bott_sal_N = tmp.bott_sal_N.values
    
    # PLOT
    title = "Mean $\Delta$ SST | "+seasons[season]+" | "+run_name
    scatter_kwargs = {"cmap":"seismic", "vmin":-1, "vmax":1, 'marker':'.', 'linewidths':0}
    f,a = plot_util.geo_scatter(lon[surf_tem_N>N_crit], lat[surf_tem_N>N_crit], 
                                c = surf_error_tem[tmp.surf_tem_N>N_crit], s=10, 
                                scatter_kwargs=scatter_kwargs, title = title)
    fn = 'smooth_surf_error_tem_'+seasons[season]+file_type
    fn_out = os.path.join(dn_output_figs, fn)
    f.savefig(fn_out)
    plt.close()
    
    
    title = "Mean $\Delta$ SSS | "+seasons[season]+" | "+run_name
    scatter_kwargs = {"cmap":"seismic", "vmin":-1, "vmax":1, 'marker':'.', 'linewidths':0}
    f,a = plot_util.geo_scatter(lon[surf_sal_N>N_crit], lat[surf_sal_N>N_crit], 
                                c = surf_error_sal[surf_sal_N>N_crit], s=10, scatter_kwargs=scatter_kwargs, title = title)
    fn = 'smooth_surf_error_sal_'+seasons[season]+file_type
    fn_out = os.path.join(dn_output_figs, fn)
    f.savefig(fn_out)
    plt.close()
    
    
    title = "Mean $\Delta$ SBT | "+seasons[season]+" | "+run_name
    scatter_kwargs = {"cmap":"seismic", "vmin":-1, "vmax":1, 'marker':'.', 'linewidths':0}
    f,a = plot_util.geo_scatter(lon[bott_tem_N>N_crit], lat[bott_tem_N>N_crit], 
                                c = bott_error_tem[bott_tem_N>N_crit], s=10, scatter_kwargs=scatter_kwargs, title = title)
    fn = 'smooth_bott_error_tem_'+seasons[season]+file_type
    fn_out = os.path.join(dn_output_figs, fn)
    f.savefig(fn_out)
    plt.close()
    
    
    title = "Mean $\Delta$ SBS | "+seasons[season]+" | "+run_name
    scatter_kwargs = {"cmap":"seismic", "vmin":-1, "vmax":1, 'marker':'.', 'linewidths':0}
    f,a = plot_util.geo_scatter(lon[bott_sal_N>N_crit], lat[bott_sal_N>N_crit], 
                                c = bott_error_sal[bott_sal_N>N_crit], s=10, scatter_kwargs=scatter_kwargs, title = title)
    fn = 'smooth_bott_error_sal_'+seasons[season]+file_type
    fn_out = os.path.join(dn_output_figs, fn)
    f.savefig(fn_out)
    plt.close()
    
def geo_scatter_profiles(var, title, fn_out, sf=3):
    cmax = np.nanmean(var) + sf*np.nanstd(var)
    cmin = np.nanmean(var) - sf*np.nanstd(var)
    
    sca = plot_util.geo_scatter(var.longitude, var.latitude, 
                c = var, s=10, scatter_kwargs=scatter_kwargs, title = title)
    print("  >>>>>  Saving: " + fn_out)
    plt.savefig(fn_out)
    plt.close()
    return