import xarray as xr
import xarray.ufuncs as uf
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/users/dbyrne/code/COAsT')
import coast
import coast.general_utils as gu
import coast.crps_util as cu

class analyse_ts_hourly_en4():
    
    def __init__(fn_nemo_data, fn_nemo_domain, fn_en4, fn_out, surface_def=5, 
                 regional_masks=None, nemo_chunks={'time_counter':50}):
        
        nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain, multiple=True, chunks=nemo_chunks)
        nemo_mask = nemo.dataset.bottom_level == 0
        nemo.dataset = nemo.dataset.rename({'t_dim':'time'})
        
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
        
        # Estimate EN4 SST as mean of top levels
        surface_ind = en4.dataset.depth > surface_def
        
        sst_en4 = en4.dataset.temperature.where(surface_ind, np.nan)
        sss_en4 = en4.dataset.practical_salinity.where(surface_ind, np.nan)
        
        sst_en4 = sst_en4.mean(dim="z_dim", skipna=True)
        sss_en4 = sss_en4.mean(dim="z_dim", skipna=True)
        
        sst_en4.load()
        sss_en4.load()
        
        ind2D = gu.nearest_indices_2D(nemo.dataset.longitude, nemo.dataset.latitude, 
                                      en4.dataset.longitude, en4.dataset.latitude,
                                      mask=nemo_mask)
        
        # For every EN4 profile, determine the nearest model time index
        # If more than t_crit away from nearest, then discard it
        n_prof = en4.dataset.dims['profile']
        
        # Estimate Model SST as TOP LEVEL (from shelftmb files)
        # nemo_surf_pts = nemo_surf.isel(x_dim = ind2D[0], y_dim = ind2D[1]).dataset
        # nemo_surf_pts = nemo_surf_pts.interp(time = en4.dataset.time.values, method='nearest')
        # tmp = xr.DataArray(np.arange(0,220739))
        # nemo_surf_pts = nemo_surf_pts.isel(dim_0 = tmp, time = tmp)
        
        sst_e = np.zeros(n_prof)*np.nan
        sss_e = np.zeros(n_prof)*np.nan
        crps_tem_1 = np.zeros(n_prof)*np.nan
        crps_sal_1 = np.zeros(n_prof)*np.nan
        crps_tem_2 = np.zeros(n_prof)*np.nan
        crps_sal_2 = np.zeros(n_prof)*np.nan
        crps_tem_3 = np.zeros(n_prof)*np.nan
        crps_sal_3 = np.zeros(n_prof)*np.nan
        
        # CRPS
        nemo.dataset = nemo.dataset[['votemper_top','vosaline_top']]
        n_nemo_time = nemo.dataset.dims['time']
        nemo_time = nemo.dataset.time.load()
        
        x_dim_len = nemo.dataset.dims['x_dim']
        y_dim_len = nemo.dataset.dims['y_dim']
        
        for tii in range(0, n_nemo_time):
            
            print(tii)
            tmp = nemo.isel(time = tii).dataset
            time_diff = np.abs( tmp.time.values - sst_en4.time.values ).astype('timedelta64[m]')
            use_ind = np.where( time_diff.astype(int) < 30 )[0]
            n_use = len(use_ind)
            
            if n_use>0:
                
                tmp.load()
                x_tmp = ind2D[0][use_ind]
                y_tmp = ind2D[1][use_ind]
                
                x_tmp = xr.where(x_tmp<x_dim_len-3, x_tmp, np.nan)
                y_tmp = xr.where(y_tmp<y_dim_len-3, y_tmp, np.nan)

                x_tmp = xr.where(x_tmp>3, x_tmp, np.nan)
                y_tmp = xr.where(y_tmp>3, y_tmp, np.nan)
                
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
                
                nh_x = [np.arange( x_tmp[ii]-1, x_tmp[ii]+2 ) for ii in range(0,n_use)] 
                nh_y = [np.arange( y_tmp[ii]-1, y_tmp[ii]+2 ) for ii in range(0,n_use)]   
                nh = [tmp.isel(x_dim = nh_x[ii], y_dim = nh_y[ii]) for ii in range(0,n_use)] 
                crps_tem_tmp = [ cu.crps_empirical(nh[ii].votemper_top.values.flatten(), sst_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_sal_tmp = [ cu.crps_empirical(nh[ii].vosaline_top.values.flatten(), sss_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_tem_1[use_ind] = crps_tem_tmp
                crps_sal_1[use_ind] = crps_sal_tmp
                
                nh_x = [np.arange( x_tmp[ii]-2, x_tmp[ii]+3 ) for ii in range(0,n_use)] 
                nh_y = [np.arange( y_tmp[ii]-2, y_tmp[ii]+3 ) for ii in range(0,n_use)]   
                nh = [tmp.isel(x_dim = nh_x[ii], y_dim = nh_y[ii]) for ii in range(0,n_use)] 
                crps_tem_tmp = [ cu.crps_empirical(nh[ii].votemper_top.values.flatten(), sst_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_sal_tmp = [ cu.crps_empirical(nh[ii].vosaline_top.values.flatten(), sss_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_tem_2[use_ind] = crps_tem_tmp
                crps_sal_2[use_ind] = crps_sal_tmp
                
                nh_x = [np.arange( x_tmp[ii]-3, x_tmp[ii]+4 ) for ii in range(0,n_use)] 
                nh_y = [np.arange( y_tmp[ii]-3, y_tmp[ii]+4 ) for ii in range(0,n_use)]   
                nh = [tmp.isel(x_dim = nh_x[ii], y_dim = nh_y[ii]) for ii in range(0,n_use)] 
                crps_tem_tmp = [ cu.crps_empirical(nh[ii].votemper_top.values.flatten(), sst_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_sal_tmp = [ cu.crps_empirical(nh[ii].vosaline_top.values.flatten(), sss_en4_tmp[ii]) for ii in range(0,n_use)]
                crps_tem_3[use_ind] = crps_tem_tmp
                crps_sal_3[use_ind] = crps_sal_tmp
                
        sst_ae = np.abs(sst_e)
        sss_ae = np.abs(sss_e)
        
        # Put everything into xarray dataset
        en4_season = get_season_index(sst_en4.time.values)
        
        # Regional Means
        n_regions = len(regional_masks)
        n_season = 4
        
        reg_array = np.zeros((n_regions, n_season))*np.nan
        
        ds = xr.Dataset(coords = dict(
                            longitude = ("profile", sst_en4.longitude.values),
                            latitude = ("profile", sst_en4.latitude.values),
                            time = ("profile", sst_en4.time.values),
                            season_ind = ("profile", en4_season)),
                        data_vars = dict(
                            sst_err = ("profile", sst_e),
                            sss_err = ("profile", sss_e),
                            sst_abs_err = ("profile", sst_ae),
                            sss_abs_err = ("profile", sss_ae),
                            sst_crps1 = ("profile", crps_tem_1),
                            sss_crps1 = ("profile", crps_sal_1),
                            sst_crps2 = ("profile", crps_tem_2),
                            sss_crps2 = ("profile", crps_sal_2),
                            sst_crps3 = ("profile", crps_tem_3),
                            sss_crps3 = ("profile", crps_sal_3)))
        
        ds_mean = xr.Dataset(coords = dict(
                            longitude = ("profile", sst_en4.longitude.values),
                            latitude = ("profile", sst_en4.latitude.values),
                            time = ("profile", sst_en4.time.values),
                            season_ind = ("profile", en4_season)),
                        data_vars = dict(
                            sst_me = (["region", "season"], reg_array.copy()),
                            sss_me = (["region", "season"], reg_array.copy()),
                            sst_mae = (["region", "season"], reg_array.copy()),
                            sss_mae = (["region", "season"], reg_array.copy()),
                            sst_crps1_mean = (["region", "season"], reg_array.copy()),
                            sss_crps1_mean = (["region", "season"], reg_array.copy()),
                            sst_crps2_mean = (["region", "season"], reg_array.copy()),
                            sss_crps2_mean = (["region", "season"], reg_array.copy()),
                            sst_crps3_mean = (["region", "season"], reg_array.copy()),
                            sss_crps3_mean = (["region", "season"], reg_array.copy())))
        
        
        for reg in range(0,n_regions):
            reg_mask = regional_masks[reg]
            reg_ind = np.where( reg_mask[ind2D[1], ind2D[0]] )[0]
            ds_reg = ds.isel(profile = reg_ind)
            ds_reg_group = ds_reg.groupby('time.season')
            ds_reg_mean = ds_reg_group.mean(skipna=True)
            ds_reg_std = ds_reg_group.std(skipna=True)
            ds_mean['sst_me'][reg, :] = ds_reg_mean.sst_err.values
            ds_mean['sss_me'][reg, :] = ds_reg_mean.sss_err.values
            ds_mean['sst_mae'][reg, :] = ds_reg_mean.sst_abs_err.values
            ds_mean['sss_mae'][reg, :] = ds_reg_mean.sss_abs_err.values
            ds_mean['sst_crps1_mean'][reg, :] = ds_reg_mean.sst_crps1.values
            ds_mean['sss_crps1_mean'][reg, :] = ds_reg_mean.sss_crps1.values
            ds_mean['sst_crps2_mean'][reg, :] = ds_reg_mean.sst_crps2.values
            ds_mean['sss_crps2_mean'][reg, :] = ds_reg_mean.sss_crps2.values
            ds_mean['sst_crps3_mean'][reg, :] = ds_reg_mean.sst_crps3.values
            ds_mean['sss_crps3_mean'][reg, :] = ds_reg_mean.sss_crps3.values
            
        ds_out = xr.merge((ds, ds_mean))
        
        # Write to file
        ds_out.to_netcdf(fn_out)
        
        return nemo, en4
    
    @staticmethod
    def get_season_index(dt):
        month_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                         7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
        dt = pd.to_datetime(dt)
        month_index = dt.month
        season_index = [month_season_dict[mm] for mm in month_index]
        return season_index
        
        
