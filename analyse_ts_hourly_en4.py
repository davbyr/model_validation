import xarray as xr
import xarray.ufuncs as uf
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/users/dbyrne/code/COAsT')
import coast
import coast.general_utils as gu
import coast.crps_util as cu
import os
import os.path
from datetime import datetime, timedelta

def get_season_index(dt):
    month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                     7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
    dt = pd.to_datetime(dt)
    month_index = dt.month
    season_index = [month_season_dict[mm] for mm in month_index]
    return season_index

def write_ds_to_file(ds, fn, **kwargs):
    if os.path.exists(fn):
        os.remove(fn)
    ds.to_netcdf(fn, **kwargs)

class analyse_ts_hourly_en4():
    
    def __init__(self, fn_nemo_data, fn_nemo_domain, fn_en4, fn_out, 
                 surface_def=2, bottom_def=10,
                 regional_masks=[], region_names=[], 
                 nemo_chunks={'time_counter':50},
                 bathymetry = None):
        
        print('0', flush=True)
        
        nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain, multiple=True, chunks=nemo_chunks)
        nemo_mask = nemo.dataset.bottom_level == 0
        nemo.dataset = nemo.dataset.rename({'t_dim':'time'})
        if bathymetry is not None:
            nemo.dataset = nemo.dataset[['votemper_top','vosaline_top',
                                         'votemper_bot','vosaline_bot']]
        else:
            nemo.dataset = nemo.dataset[['votemper_top','vosaline_top']]
        
        print('a', flush=True)
        
        en4 = coast.PROFILE()
        en4.read_EN4(fn_en4, multiple=True)
        
        # Get obs in box
        lonmax = np.nanmax(nemo.dataset['longitude'])
        lonmin = np.nanmin(nemo.dataset['longitude'])
        latmax = np.nanmax(nemo.dataset['latitude'])
        latmin = np.nanmin(nemo.dataset['latitude'])
        ind = coast.general_utils.subset_indices_lonlat_box(en4.dataset['longitude'], 
                                                            en4.dataset['latitude'],
                                                            lonmin, lonmax, 
                                                            latmin, latmax)[0]
        en4 = en4.isel(profile=ind)
        print('b', flush=True)
        
        # Get obs time slice
        n_nemo_time = nemo.dataset.dims['time']
        nemo.dataset.time.load()
        en4.dataset.time.load()
        nemo_time = pd.to_datetime(nemo.dataset.time.values)
        en4_time = pd.to_datetime(en4.dataset.time.values)
        time_max = max(nemo_time) + timedelta(hours=1)
        time_min = min(nemo_time) - timedelta(hours=1)
        time_ind0 = en4_time <= time_max
        time_ind1 = en4_time >= time_min
        time_ind = np.logical_and(time_ind0, time_ind1)
        en4 = en4.isel(profile=time_ind)
        
        # Get model indices
        en4_time = pd.to_datetime(en4.dataset.time.values)
        ind2D = gu.nearest_indices_2D(nemo.dataset.longitude, nemo.dataset.latitude, 
                                      en4.dataset.longitude, en4.dataset.latitude,
                                      mask=nemo_mask)
        
        print('c', flush=True)
        
        # Estimate EN4 SST as mean of top levels
        surface_ind = en4.dataset.depth <= surface_def
        
        sst_en4 = en4.dataset.potential_temperature.where(surface_ind, np.nan)
        sss_en4 = en4.dataset.practical_salinity.where(surface_ind, np.nan)
        
        sst_en4 = sst_en4.mean(dim="z_dim", skipna=True).load()
        sss_en4 = sss_en4.mean(dim="z_dim", skipna=True).load()
        
        print('d', flush=True)
        
        # Bottom values
        if bathymetry is not None:
            bathy_pts = bathymetry.isel(x_dim = ind2D[0], y_dim = ind2D[1]).swap_dims({'dim_0':'profile'})
            bottom_ind = en4.dataset.depth >= (bathy_pts - bottom_def)

            sbt_en4 = en4.dataset.potential_temperature.where(bottom_ind, np.nan)
            sbs_en4 = en4.dataset.practical_salinity.where(bottom_ind, np.nan)
        
            sbt_en4 = sbt_en4.mean(dim="z_dim", skipna=True).load()
            sbs_en4 = sbs_en4.mean(dim="z_dim", skipna=True).load()
        
        print('e', flush=True)
        
        # For every EN4 profile, determine the nearest model time index
        # If more than t_crit away from nearest, then discard it
        n_prof = en4.dataset.dims['profile']
        
        sst_e = np.zeros(n_prof)*np.nan
        sss_e = np.zeros(n_prof)*np.nan
        sst_ae = np.zeros(n_prof)*np.nan
        sss_ae = np.zeros(n_prof)*np.nan
        crps_tem_2 = np.zeros(n_prof)*np.nan
        crps_sal_2 = np.zeros(n_prof)*np.nan
        crps_tem_4 = np.zeros(n_prof)*np.nan
        crps_sal_4 = np.zeros(n_prof)*np.nan
        crps_tem_6 = np.zeros(n_prof)*np.nan
        crps_sal_6 = np.zeros(n_prof)*np.nan
        
        sbt_e = np.zeros(n_prof)*np.nan
        sbs_e = np.zeros(n_prof)*np.nan
        sbt_e = np.zeros(n_prof)*np.nan
        sbs_e = np.zeros(n_prof)*np.nan
        
        # CRPS
        
        x_dim_len = nemo.dataset.dims['x_dim']
        y_dim_len = nemo.dataset.dims['y_dim']
        
        n_r = nemo.dataset.dims['y_dim']
        n_c = nemo.dataset.dims['x_dim']
        regional_masks = regional_masks.copy()
        region_names = region_names.copy()
        regional_masks.append(np.ones((n_r, n_c)))
        region_names.append('whole_domain')
        n_regions = len(regional_masks)
        n_season = 5
        
        print('Starting analysis')
        
        for tii in range(0, n_nemo_time):
            
            print(nemo_time[tii], flush=True)
            
            time_diff = np.abs( nemo_time[tii] - en4_time ).astype('timedelta64[m]')
            use_ind = np.where( time_diff.astype(int) < 30 )[0]
            n_use = len(use_ind)
            
            if n_use>0:
                
                tmp = nemo.isel(time = tii).dataset
                tmp.load()
                x_tmp = ind2D[0][use_ind]
                y_tmp = ind2D[1][use_ind]
                
                x_tmp = xr.where(x_tmp<x_dim_len-7, x_tmp, np.nan)
                y_tmp = xr.where(y_tmp<y_dim_len-7, y_tmp, np.nan)

                x_tmp = xr.where(x_tmp>7, x_tmp, np.nan)
                y_tmp = xr.where(y_tmp>7, y_tmp, np.nan)
                
                shared_mask = np.logical_or(np.isnan(x_tmp), np.isnan(y_tmp))
                shared_mask = np.where(~shared_mask)
                
                x_tmp = x_tmp[shared_mask].astype(int)
                y_tmp = y_tmp[shared_mask].astype(int)
                use_ind = use_ind[shared_mask].astype(int)
                
                n_use = len(use_ind)
                if n_use<1:
                    continue
                
                tmp_pts = tmp.isel(x_dim = x_tmp, y_dim = y_tmp)
                sst_en4_tmp = sst_en4.values[use_ind]
                sss_en4_tmp = sss_en4.values[use_ind]
                sst_e[use_ind] = tmp_pts.votemper_top.values - sst_en4_tmp
                sss_e[use_ind] = tmp_pts.vosaline_top.values - sss_en4_tmp
                
                if bathymetry is not None:
                    sbt_en4_tmp = sbt_en4.values[use_ind]
                    sbs_en4_tmp = sbs_en4.values[use_ind]
                    sbt_e[use_ind] = tmp_pts.votemper_bot.values - sbt_en4_tmp
                    sbs_e[use_ind] = tmp_pts.vosaline_bot.values - sbs_en4_tmp
                
                nh_x = [np.arange( x_tmp[ii]-2, x_tmp[ii]+3 ) for ii in range(0,n_use)] 
                nh_y = [np.arange( y_tmp[ii]-2, y_tmp[ii]+3 ) for ii in range(0,n_use)]   
                nh = [tmp.isel(x_dim = nh_x[ii], y_dim = nh_y[ii]) for ii in range(0,n_use)] 
                crps_tem_tmp = [ cu.crps_empirical(nh[ii].votemper_top.values.flatten(), sst_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_sal_tmp = [ cu.crps_empirical(nh[ii].vosaline_top.values.flatten(), sss_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_tem_2[use_ind] = crps_tem_tmp
                crps_sal_2[use_ind] = crps_sal_tmp
                
                nh_x = [np.arange( x_tmp[ii]-4, x_tmp[ii]+5 ) for ii in range(0,n_use)] 
                nh_y = [np.arange( y_tmp[ii]-4, y_tmp[ii]+5 ) for ii in range(0,n_use)]   
                nh = [tmp.isel(x_dim = nh_x[ii], y_dim = nh_y[ii]) for ii in range(0,n_use)] 
                crps_tem_tmp = [ cu.crps_empirical(nh[ii].votemper_top.values.flatten(), sst_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_sal_tmp = [ cu.crps_empirical(nh[ii].vosaline_top.values.flatten(), sss_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_tem_4[use_ind] = crps_tem_tmp
                crps_sal_4[use_ind] = crps_sal_tmp
                
                nh_x = [np.arange( x_tmp[ii]-6, x_tmp[ii]+7 ) for ii in range(0,n_use)] 
                nh_y = [np.arange( y_tmp[ii]-6, y_tmp[ii]+7 ) for ii in range(0,n_use)]   
                nh = [tmp.isel(x_dim = nh_x[ii], y_dim = nh_y[ii]) for ii in range(0,n_use)] 
                crps_tem_tmp = [ cu.crps_empirical(nh[ii].votemper_top.values.flatten(), sst_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_sal_tmp = [ cu.crps_empirical(nh[ii].vosaline_top.values.flatten(), sss_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_tem_6[use_ind] = crps_tem_tmp
                crps_sal_6[use_ind] = crps_sal_tmp
                    
        print('Profile analysis done', flush=True)
        sst_ae = np.abs(sst_e)
        sss_ae = np.abs(sss_e)
        sbt_ae = np.abs(sbt_e)
        sbs_ae = np.abs(sbs_e)
        # Put everything into xarray dataset
        en4_season = get_season_index(sst_en4.time.values)
        
        # Regional Means        
        reg_array = np.zeros((n_regions, n_season))*np.nan
        is_in_region = [mm[ind2D[1], ind2D[0]] for mm in regional_masks]
        is_in_region = np.array(is_in_region, dtype=bool)
        
        ds = xr.Dataset(coords = dict(
                            longitude = ("profile", sst_en4.longitude.values),
                            latitude = ("profile", sst_en4.latitude.values),
                            time = ("profile", sst_en4.time.values),
                            season_ind = ("profile", en4_season)),
                        data_vars = dict(
                            obs_sst = ('profile', sst_en4.values),
                            obs_sss = ('profile', sss_en4.values),
                            sst_err = ("profile", sst_e),
                            sss_err = ("profile", sss_e),
                            sst_abs_err = ("profile", sst_ae),
                            sss_abs_err = ("profile", sss_ae),
                            sst_crps2 = ("profile", crps_tem_2),
                            sss_crps2 = ("profile", crps_sal_2),
                            sst_crps4 = ("profile", crps_tem_4),
                            sss_crps4 = ("profile", crps_sal_4),
                            sst_crps6 = ("profile", crps_tem_6),
                            sss_crps6 = ("profile", crps_sal_6)))
        
        season_names = ['All','DJF','MAM','JJA','SON']
        ds = ds.chunk({'profile':10000})
        
        ds_mean = xr.Dataset(coords = dict(
                            longitude = ("profile", sst_en4.longitude.values),
                            latitude = ("profile", sst_en4.latitude.values),
                            time = ("profile", sst_en4.time.values),
                            season_ind = ("profile", en4_season),
                            region_names = ('region', region_names),
                            season = ('season', season_names)),
                        data_vars = dict(
                            sst_me = (["region", "season"],  reg_array.copy()),
                            sss_me = (["region", "season"],  reg_array.copy()),
                            sst_mae = (["region", "season"], reg_array.copy()),
                            sss_mae = (["region", "season"], reg_array.copy()),
                            sst_crps2_mean = (["region", "season"], reg_array.copy()),
                            sss_crps2_mean = (["region", "season"], reg_array.copy()),
                            sst_crps4_mean = (["region", "season"], reg_array.copy()),
                            sss_crps4_mean = (["region", "season"], reg_array.copy()),
                            sst_crps6_mean = (["region", "season"], reg_array.copy()),
                            sss_crps6_mean = (["region", "season"], reg_array.copy())))
        
        if bathymetry is not None:
            ds_mean['sbt_me'] = (['region','season'], reg_array.copy())
            ds_mean['sbs_me'] = (['region','season'], reg_array.copy())
            ds_mean['sbt_mae'] = (['region','season'], reg_array.copy())
            ds_mean['sbs_mae'] = (['region','season'], reg_array.copy())
                            
            ds['obs_sbt'] = (['profile'], sbt_en4.values)
            ds['obs_sbs'] = (['profile'], sbs_en4.values)
            ds['sbt_err'] = (['profile'], sbt_e)
            ds['sbs_err'] = (['profile'], sbs_e)
            ds['sbt_abs_err'] = (['profile'], sbt_ae)
            ds['sbs_abs_err'] = (['profile'], sbs_ae)
                                              
        
        
        for reg in range(0,n_regions):
            reg_ind = np.where( is_in_region[reg].astype(bool) )[0]
            ds_reg = ds.isel(profile = reg_ind)
            ds_reg_group = ds_reg.groupby('time.season')
            ds_reg_mean = ds_reg_group.mean(skipna=True).compute()
            
            ds_mean['sst_me'][reg, 1:]  = ds_reg_mean.sst_err.values
            ds_mean['sss_me'][reg, 1:]  = ds_reg_mean.sss_err.values
            ds_mean['sst_mae'][reg, 1:] = ds_reg_mean.sst_abs_err.values
            ds_mean['sss_mae'][reg, 1:] = ds_reg_mean.sss_abs_err.values
            ds_mean['sst_crps2_mean'][reg, 1:] = ds_reg_mean.sst_crps2.values
            ds_mean['sss_crps2_mean'][reg, 1:] = ds_reg_mean.sss_crps2.values
            ds_mean['sst_crps4_mean'][reg, 1:] = ds_reg_mean.sst_crps4.values
            ds_mean['sss_crps4_mean'][reg, 1:] = ds_reg_mean.sss_crps4.values
            ds_mean['sst_crps6_mean'][reg, 1:] = ds_reg_mean.sst_crps6.values
            ds_mean['sss_crps6_mean'][reg, 1:] = ds_reg_mean.sss_crps6.values
            
            if bathymetry is not None:
                ds_mean['sbt_me'][reg, 1:]  = ds_reg_mean.sbt_err.values
                ds_mean['sbs_me'][reg, 1:]  = ds_reg_mean.sbs_err.values
                ds_mean['sbt_mae'][reg, 1:] = ds_reg_mean.sbt_abs_err.values
                ds_mean['sbs_mae'][reg, 1:] = ds_reg_mean.sbs_abs_err.values
            
            ds_reg_mean = ds_reg.mean(dim='profile', skipna=True).compute()
            ds_mean['sst_me'][reg, 0]  = ds_reg_mean.sst_err.values
            ds_mean['sss_me'][reg, 0]  = ds_reg_mean.sss_err.values
            ds_mean['sst_mae'][reg, 0] = ds_reg_mean.sst_abs_err.values
            ds_mean['sss_mae'][reg, 0] = ds_reg_mean.sss_abs_err.values
            ds_mean['sst_crps2_mean'][reg, 0] = ds_reg_mean.sst_crps2.values
            ds_mean['sss_crps2_mean'][reg, 0] = ds_reg_mean.sss_crps2.values
            ds_mean['sst_crps4_mean'][reg, 0] = ds_reg_mean.sst_crps4.values
            ds_mean['sss_crps4_mean'][reg, 0] = ds_reg_mean.sss_crps4.values
            ds_mean['sst_crps6_mean'][reg, 0] = ds_reg_mean.sst_crps6.values
            ds_mean['sss_crps6_mean'][reg, 0] = ds_reg_mean.sss_crps6.values
            
            if bathymetry is not None:
                ds_mean['sbt_me'][reg, 0]  = ds_reg_mean.sbt_err.values
                ds_mean['sbs_me'][reg, 0]  = ds_reg_mean.sbs_err.values
                ds_mean['sbt_mae'][reg, 0] = ds_reg_mean.sbt_abs_err.values
                ds_mean['sbs_mae'][reg, 0] = ds_reg_mean.sbs_abs_err.values
                
            
        ds_out = xr.merge((ds, ds_mean))
        ds_out['is_in_region'] = (['region','profile'], is_in_region)
        
        # Write to file
        write_ds_to_file(ds_out, fn_out)
        
        
