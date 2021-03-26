import matplotlib.pyplot as plt
import coast
import coast.general_utils as gu
import numpy as np
import coast.plot_util as plot_util
import xarray as xr
import os.path

fn_profile_stats = "/Users/dbyrne/Projects/CO9_AMM15/data/analysis/p0/en4_stats_by_profile*" 
dn_output_figs = "/Users/dbyrne/Projects/CO9_AMM15/data/figs/p0"

stats_profile = xr.open_mfdataset(fn_profile_stats, chunks={})
radius = 25
run_name = 'CO9_AMM15p0'
seasons = ['Annual','DJF','MAM','JJA','SON']

for season in range(1,5):
    ind_season = stats_profile.season == season
    tmp = stats_profile.isel(profile=ind_season)
    tmp = tmp[['surf_error_tem', 'surf_error_sal','bott_error_tem','bott_error_sal']]
    tmp.load()
    lon1 = np.arange(-20, 12, 0.25)
    lat1 = np.arange(44, 64, 0.25)
    
    lon2, lat2 = np.meshgrid(lon1, lat1)
    
    lon = lon2.flatten()
    lat = lat2.flatten()
    
    tem_error = np.zeros(len(lon))
    sal_error = np.zeros(len(lon))
    tem_N = np.zeros(len(lon))
    sal_N = np.zeros(len(lon))
    
    for ii in range(0,len(lon)): 
        ind = gu.subset_indices_by_distance(tmp.longitude, tmp.latitude, 
                                            lon[ii], lat[ii], radius=radius)
        tem = tmp.surf_error_tem.isel(profile=ind).values
        sal = tmp.surf_error_sal.isel(profile=ind).values
        tem_error[ii] = np.nanmean(tem)
        sal_error[ii] = np.nanmean(sal)
        tem_N[ii] = tem_N[ii] + np.sum( ~np.isnan(tem) )
        sal_N[ii] = sal_N[ii] + np.sum( ~np.isnan(sal) )
        
    N_crit =5
    
    # PLOT
    title = "Mean $\Delta$ SST | "+seasons[season]+" | "+run_name
    scatter_kwargs = {"cmap":"seismic", "vmin":-1, "vmax":1, 'marker':'.', 'linewidths':0}
    f,a = plot_util.geo_scatter(lon[tem_N>N_crit], lat[tem_N>N_crit], 
                                c = tem_error[tem_N>N_crit], s=10, scatter_kwargs=scatter_kwargs, title = title)
    fn = 'smooth_surf_error_tem_'+seasons[season]+'.eps'
    fn_out = os.path.join(dn_output_figs, fn)
    f.savefig(fn_out)
    
    
    title = "Mean $\Delta$ SSS | "+seasons[season]+" | "+run_name
    scatter_kwargs = {"cmap":"seismic", "vmin":-1, "vmax":1, 'marker':'.', 'linewidths':0}
    f,a = plot_util.geo_scatter(lon[tem_N>N_crit], lat[tem_N>N_crit], c = sal_error[tem_N>N_crit], s=10, scatter_kwargs=scatter_kwargs, title = title)
    fn = 'smooth_surf_error_sal_'+seasons[season]+'.eps'
    fn_out = os.path.join(dn_output_figs, fn)
    f.savefig(fn_out)
    
def geo_scatter_profiles(var, title, fn_out, sf=3):
    cmax = np.nanmean(var) + sf*np.nanstd(var)
    cmin = np.nanmean(var) - sf*np.nanstd(var)
    
    sca = plot_util.geo_scatter(var.longitude, var.latitude, 
                c = var, s=10, scatter_kwargs=scatter_kwargs, title = title)
    print("  >>>>>  Saving: " + fn_out)
    plt.savefig(fn_out)
    plt.close()
    return