import matplotlib.pyplot as plt
import coast
import coast.general_utils as gu
import numpy as np
import coast.plot_util as plot_util
import xarray as xr
import os.path

fn_profile_stats = "/Users/dbyrne/Projects/CO9_AMM15/data/analysis/co7/en4_stats_by_profile*" 
dn_output = "/Users/dbyrne/Projects/CO9_AMM15/data/analysis/co7"

stats_profile = xr.open_mfdataset(fn_profile_stats, chunks={})
radius = 25
run_name = 'CO9_AMM15p0'
seasons = ['Annual','DJF','MAM','JJA','SON']
n_seasons = len(seasons)

lon1 = np.arange(-20, 12, 0.25)
lat1 = np.arange(44, 64, 0.25)

lon2, lat2 = np.meshgrid(lon1, lat1)

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
    
    for ii in range(0,len(lon)): 
        ind = gu.subset_indices_by_distance(tmp.longitude, tmp.latitude, 
                                            lon[ii], lat[ii], radius=radius)
        
        tem = tmp.surf_error_tem.isel(profile=ind).values
        sal = tmp.surf_error_sal.isel(profile=ind).values
        surf_error_tem[season,ii] = np.nanmean(tem)
        surf_error_sal[season,ii] = np.nanmean(sal)
        surf_tem_N[season,ii] = surf_tem_N[season,ii] + np.sum( ~np.isnan(tem) )
        surf_sal_N[season,ii] = surf_sal_N[season,ii] + np.sum( ~np.isnan(sal) )
        
        tem = tmp.bott_error_tem.isel(profile=ind).values
        sal = tmp.bott_error_sal.isel(profile=ind).values
        bott_error_tem[season,ii] = np.nanmean(tem)
        bott_error_sal[season,ii] = np.nanmean(sal)
        bott_tem_N[season,ii] = bott_tem_N[season,ii] + np.sum( ~np.isnan(tem) )
        bott_sal_N[season,ii] = bott_sal_N[season,ii] + np.sum( ~np.isnan(sal) )
        
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
fn = os.path.join(dn_output,'en4_profile_stats_smoothed.nc')
ds.to_netcdf(fn)