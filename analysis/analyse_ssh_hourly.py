import sys
sys.path.append('/work/n01/n01/dbyrne/CO9_AMM15/code/COAsT')
import coast
import coast.general_utils as gu
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import utide as ut
import scipy.signal as signal
import os
import glob
from dask.diagnostics import ProgressBar

def write_stats_to_file(stats, fn_out):
    print("analyse_ssh_hourly: Writing output to file")
    if os.path.exists(fn_out):
        os.remove(fn_out)
    stats.to_netcdf(fn_out)
    print("analyse_monthly_ssh: Done")


def read_nemo_ssh(fn_nemo_data, fn_nemo_domain, chunks):
    print("analyse_ssh_hourly: Reading NEMO data")
    nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain, 
                              multiple=True, chunks=chunks).dataset
    print("analyse_ssh_hourly: Done")
    return nemo[['ssh', 'time_instant']]

def read_nemo_oneatatime(fn_nemo_data, fn_nemo_domain, obs, landmask, 
                         chunks):
    print("analyse_ssh_hourly: a")
    
    file_list = glob.glob(fn_nemo_data)
    
    file=file_list[0]
    nemo = coast.NEMO(file, fn_nemo_domain, chunks=chunks).dataset
    
    ind2D = gu.nearest_indices_2D(nemo.longitude, nemo.latitude, 
                                obs.longitude, obs.latitude,
                                mask = landmask)
    print("analyse_ssh_hourly: b")
    nemo_list = []
    
    for ff in range(0,len(file_list)):
        file = file_list[ff]
        print(file)
        nemo = coast.NEMO(file, fn_nemo_domain, chunks=chunks).dataset
        nemo = nemo['ssh']
        nemo_ext = nemo.isel(x_dim = ind2D[0], y_dim = ind2D[1]).load()
        nemo_ext = nemo_ext.swap_dims({'dim_0':'port'})
        nemo_list.append(nemo_ext)
    
    print("analyse_ssh_hourly: c")
    
    nemo = xr.merge(nemo_list)
    
    print('d')
    
    return nemo

def read_nemo_landmask_using_top_level(fn_nemo_domain):
    print("analyse_ssh_hourly: Reading landmask")
    dom = xr.open_dataset(fn_nemo_domain)
    landmask = np.array(dom.top_level.values.squeeze() == 0)
    dom.close()
    print("analyse_ssh_hourly: Done")
    return landmask

def read_obs_data(fn_obs):
    return xr.open_dataset(fn_obs)

def subset_obs_by_lonlat(nemo, obs):
    print("analyse_ssh_hourly: Subsetting obs data")
    lonmax = np.nanmax(nemo.longitude)
    lonmin = np.nanmin(nemo.longitude)
    latmax = np.nanmax(nemo.latitude)
    latmin = np.nanmin(nemo.latitude)
    ind = gu.subset_indices_lonlat_box(obs.longitude, obs.latitude, 
                                       lonmin, lonmax, latmin, latmax)
    obs = obs.isel(port=ind[0])
    print("analyse_ssh_hourly: Done")
    return obs

def extract_obs_locations(nemo, obs, landmask):
    print("analyse_ssh_hourly: Extracting nearest model points ")
    # Extract model locations
    ind2D = gu.nearest_indices_2D(nemo.longitude, nemo.latitude, 
                                obs.longitude, obs.latitude,
                                mask = landmask)
    print("analyse_ssh_hourly: determined indices, loading data")
    nemo_extracted = nemo.isel(x_dim = ind2D[0], y_dim = ind2D[1])
    nemo_extracted = nemo_extracted.swap_dims({'dim_0':'port'})
    
    with ProgressBar():
        nemo_extracted.load()
    
    # Check interpolation distances
    max_dist = 5
    interp_dist = gu.calculate_haversine_distance(nemo_extracted.longitude, 
                                                  nemo_extracted.latitude, 
                                                  obs.longitude.values,
                                                  obs.latitude.values)
    keep_ind = interp_dist < max_dist
    nemo_extracted = nemo_extracted.isel(port=keep_ind)
    obs = obs.isel(port=keep_ind)
    print("analyse_ssh_hourly: Done")
    return nemo_extracted, obs

def align_timings(nemo_extracted, obs):
    print("analyse_ssh_hourly: Aligning obs and model times")
    obs = obs.interp(time = nemo_extracted.time_instant.values, method = 'linear')
    print("analyse_ssh_hourly: Done")
    return obs

def compare_phase(g1, g2):
    g1 = np.array(g1)
    g2 = np.array(g2)
    r = (g1-g2)%360 - 360
    r[r<-180] = r[r<-180] + 360
    r[r>180] = r[r>180] - 360
    return r

class analyse_ssh_hourly():
    
    def __init__(self, fn_nemo_data, fn_nemo_domain, fn_obs, fn_out,
                         thresholds = np.arange(0,2,0.1),
                         constit_to_save = ['M2', 'S2', 'K1','O1'], 
                         chunks = {'time_counter':50}):
        
        nemo = read_nemo_ssh(fn_nemo_data, fn_nemo_domain, chunks)
        
        landmask = read_nemo_landmask_using_top_level(fn_nemo_domain)
        
        obs = read_obs_data(fn_obs)
        
        obs = subset_obs_by_lonlat(nemo, obs)
        
        nemo_extracted, obs = extract_obs_locations(nemo, obs, landmask)
        #nemo_extracted = self.read_nemo_oneatatime(fn_nemo_data, fn_nemo_domain, 
        #                                           obs, landmask, chunks)
        
        obs = align_timings(nemo_extracted, obs)
        
        # Define Dimension Sizes
        n_port = obs.dims['port']
        n_time = obs.dims['time']
        n_constit = len(constit_to_save)
        n_thresholds = len(thresholds)
        
        a_mod = np.zeros((n_port, n_constit))*np.nan
        a_obs = np.zeros((n_port, n_constit))*np.nan
        g_mod = np.zeros((n_port, n_constit))*np.nan
        g_obs = np.zeros((n_port, n_constit))*np.nan
        
        std_obs = np.zeros((n_port))*np.nan
        std_mod = np.zeros((n_port))*np.nan
        std_err = np.zeros((n_port))*np.nan
        ntr_corr = np.zeros((n_port))*np.nan
        ntr_mae = np.zeros((n_port))*np.nan
        
        skew_mod = []
        skew_obs = []
        skew_err = []
        
        thresh_freq_ntr_mod = np.zeros((n_port, n_thresholds))
        thresh_freq_ntr_obs = np.zeros((n_port, n_thresholds))
        thresh_int_ntr_mod = np.zeros((n_port, n_thresholds))
        thresh_int_ntr_obs = np.zeros((n_port, n_thresholds))
        thresh_freq_skew_mod = np.zeros((n_port, n_thresholds))
        thresh_freq_skew_obs = np.zeros((n_port, n_thresholds))
        
        ntr_mod_all = np.zeros((n_port, n_time))*np.nan
        ntr_obs_all = np.zeros((n_port, n_time))*np.nan
        
        # Loop over tide gauge locations, perform analysis per location
        for pp in range(0,n_port):
            port_mod = nemo_extracted.isel(port=pp)
            port_obs = obs.isel(port=pp)
            
            if all(np.isnan(port_obs.ssh)):
                skew_mod.append([])
                skew_obs.append([])
                continue 
            
            # Masked arrays
            ssh_mod = port_mod.ssh.values
            ssh_obs = port_obs.ssh.values
            shared_mask = np.logical_or(np.isnan(ssh_mod), np.isnan(ssh_obs))
            ssh_mod[shared_mask] = np.nan
            ssh_obs[shared_mask] = np.nan
            time_mod = port_mod.time_instant.values
            time_obs = port_obs.time.values
            
            if np.sum(~np.isnan(ssh_obs)) < 8760:
                skew_mod.append([])
                skew_obs.append([])
                continue
            
            # Harmonic analysis datenums
            hat  = mdates.date2num(time_mod)
            
            # Do harmonic analysis using UTide
            uts_obs = ut.solve(hat, ssh_obs, lat=port_obs.latitude.values)
            uts_mod = ut.solve(hat, ssh_mod, lat=port_mod.latitude.values)
            
            # Reconstruct tidal signal 
            tide_obs = np.array( ut.reconstruct(hat, uts_obs).h)
            tide_mod = np.array( ut.reconstruct(hat, uts_mod).h)
            tide_obs[shared_mask] = np.nan
            tide_mod[shared_mask] = np.nan
            
            # Identify Peaks in tide and TWL 
            
            pk_ind_tide_mod,_ = signal.find_peaks(tide_mod, distance = 9)
            pk_ind_tide_obs,_ = signal.find_peaks(tide_obs, distance = 9)
            pk_ind_ssh_mod,_  = signal.find_peaks(ssh_mod, distance = 9)
            pk_ind_ssh_obs,_  = signal.find_peaks(ssh_obs, distance = 9)
            
            pk_time_tide_mod = pd.to_datetime( time_mod[pk_ind_tide_mod] )
            pk_time_tide_obs = pd.to_datetime( time_obs[pk_ind_tide_obs] )
            pk_time_ssh_mod  = pd.to_datetime( time_mod[pk_ind_ssh_mod] )
            pk_time_ssh_obs  = pd.to_datetime( time_obs[pk_ind_ssh_obs] )
            
            pk_tide_mod = tide_mod[pk_ind_tide_mod]
            pk_tide_obs = tide_obs[pk_ind_tide_obs] 
            pk_ssh_mod  = ssh_mod[pk_ind_ssh_mod]
            pk_ssh_obs  = ssh_obs[pk_ind_ssh_obs]
            
            # Define Skew Surges
            n_tide_mod = len(pk_tide_mod)
            n_tide_obs = len(pk_tide_obs)
            
            pk_ssh_mod_interp = np.zeros(n_tide_mod)
            pk_ssh_obs_interp = np.zeros(n_tide_obs)
            
            # Model Skew Surge
            for ii in range(0, n_tide_mod):
                time_diff = np.abs(pk_time_tide_mod[ii] - pk_time_ssh_mod)
                search_ind = np.where(time_diff < timedelta(hours=6))
                if len(search_ind[0]) > 0:
                    pk_ssh_mod_interp[ii] = np.nanmax(pk_ssh_mod[search_ind[0]])
                else:
                    pk_ssh_mod_interp[ii] = np.nan
                    
            # Observed Skew Surge
            pk_ssh_obs_interp = np.zeros(n_tide_obs)
            for ii in range(0, n_tide_obs):
                time_diff = np.abs(pk_time_tide_obs[ii] - pk_time_ssh_obs)
                search_ind = np.where(time_diff < timedelta(hours=6))
                if len(search_ind[0]) > 0:
                    pk_ssh_obs_interp[ii] = np.nanmax(pk_ssh_obs[search_ind])
                else:
                    pk_ssh_obs_interp[ii] = np.nan
                    
            skew_mod_tmp = pk_ssh_mod_interp - pk_tide_mod
            skew_obs_tmp = pk_ssh_obs_interp - pk_tide_obs
            
            ds_tmp = xr.Dataset(coords = dict(
                                    time = ('time',pk_time_tide_mod)),
                                data_vars = dict(
                                    ssh = ('time',skew_mod_tmp)))
            ds_int = ds_tmp.interp(time=pk_time_tide_obs, method='nearest')
            skew_mod_tmp = ds_int.ssh.values
        
            skew_mod.append(skew_mod_tmp)
            skew_obs.append(skew_obs_tmp)
            skew_err.append(skew_mod_tmp - skew_obs_tmp)
            
            # TWL: Basic stats
            std_obs[pp] = np.nanstd(ssh_obs)
            std_mod[pp] = np.nanstd(ssh_mod)
            
            # TWL: Constituents
            a_dict_obs = dict( zip(uts_obs.name, uts_obs.A) )
            a_dict_mod = dict( zip(uts_mod.name, uts_mod.A) )
            g_dict_obs = dict( zip(uts_obs.name, uts_obs.g) )
            g_dict_mod = dict( zip(uts_mod.name, uts_mod.g) )
            
            for cc in range(0, len(constit_to_save)):
                if constit_to_save[cc] in uts_obs.name:
                    a_mod[pp,cc] = a_dict_mod[constit_to_save[cc]] 
                    a_obs[pp,cc] = a_dict_obs[constit_to_save[cc]] 
                    g_mod[pp,cc] = g_dict_mod[constit_to_save[cc]] 
                    g_obs[pp,cc] = g_dict_obs[constit_to_save[cc]]
            
            a_mod[a_mod==0] = np.nan
            a_mod[a_mod>20] = np.nan
            a_obs[a_obs==0] = np.nan
            a_obs[a_obs>20] = np.nan
            
            # NTR: Calculate and get peaks
            ntr_obs = ssh_obs - tide_obs
            ntr_mod = ssh_mod - tide_mod
            
            ntr_obs = np.ma.masked_invalid(ntr_obs)
            ntr_mod = np.ma.masked_invalid(ntr_mod)
            
            ntr_obs = signal.savgol_filter(ntr_obs,25,3)
            ntr_mod = signal.savgol_filter(ntr_mod,25,3)
            
            ntr_obs_all[pp] = ntr_obs
            ntr_mod_all[pp] = ntr_mod
            
            pk_ind_ntr_obs,_ = signal.find_peaks(ntr_obs, distance = 12)
            pk_ind_ntr_mod,_ = signal.find_peaks(ntr_mod, distance = 12)
            
            pk_time_ntr_obs = pd.to_datetime( time_obs[pk_ind_ntr_obs] )
            pk_time_ntr_mod = pd.to_datetime( time_mod[pk_ind_ntr_mod] )
            pk_ntr_obs = ntr_obs[pk_ind_ntr_obs]
            pk_ntr_mod = ntr_mod[pk_ind_ntr_mod]
            
            # NTR: Basic stats
            ntr_corr[pp] = np.ma.corrcoef(ntr_obs, ntr_mod)[1,0]
            ntr_mae[pp] = np.ma.mean( np.abs( ntr_obs - ntr_mod) )
            
            
            # Threshold Statistics
            for nn in range(0,n_thresholds):
                threshn = thresholds[nn]
                # NTR: Threshold Frequency (Peaks)
                thresh_freq_ntr_mod[pp, nn] = np.sum( pk_ntr_mod > threshn)
                thresh_freq_ntr_obs[pp, nn] = np.sum( pk_ntr_obs > threshn)
                
                # NTR: Threshold integral (Time over threshold)
                thresh_int_ntr_mod[pp, nn] = np.sum( ntr_mod > threshn)
                thresh_int_ntr_obs[pp, nn] = np.sum( ntr_obs > threshn)
                
                # NTR: Extreme Value Analysis
                
                # Skew Surge Threshold Frequency
                thresh_freq_skew_mod[pp, nn] = np.sum( skew_mod_tmp > threshn)
                thresh_freq_skew_obs[pp, nn] = np.sum( skew_obs_tmp > threshn)
                
                # Skew surge Extreme Value Analysis
                
        # NTR: Monthly Variability
        ds_ntr = xr.Dataset(coords = dict(
                                time = ('time', obs.time.values)),
                            data_vars = dict(
                                ntr_mod = (['port','time'], ntr_mod_all),
                                ntr_obs = (['port','time'], ntr_obs_all)))
        
        # NTR: Monthly Climatology
        ntr_grouped = ds_ntr.groupby('time.month')
        ntr_clim_var = ntr_grouped.var()
        ntr_clim_mean = ntr_grouped.mean()
        
        # NTR: Monthly Means
        ntr_resampled = ds_ntr.resample(time='1M')
        ntr_monthly_var = ntr_resampled.var()
        ntr_monthly_mean = ntr_resampled.mean()
        ntr_monthly_max = ntr_resampled.max()
        
        ### Put into Dataset and write to file
        
        # Figure out skew surge dimensions
        n_skew = 0
        for pp in range(0,n_port):
            if len(skew_mod[pp])>n_skew:
                n_skew=len(skew_mod[pp])
            if len(skew_obs[pp])>n_skew:
                n_skew=len(skew_obs[pp])
                
        skew_mod_np = np.zeros((n_port, n_skew))*np.nan
        skew_obs_np = np.zeros((n_port, n_skew))*np.nan
        
        for pp in range(0, n_port):
            len_mod = len(skew_mod[pp])
            len_obs = len(skew_obs[pp])
            skew_mod_np[pp, :len_mod] = skew_mod[pp] 
            skew_obs_np[pp, :len_obs] = skew_obs[pp]
                
        stats = xr.Dataset(coords = dict(
                        longitude = ('port', obs.longitude.values),
                        latitude = ('port', obs.latitude.values),
                        time = ('time', time_obs),
                        constituent = ('constituent', constit_to_save),
                        threshold = ('threshold', thresholds),
                        time_month = ('time_month', ntr_monthly_var.time),
                        clim_month = ('clim_month', ntr_clim_var.month)),
                   data_vars = dict(
                        ssh_mod = (['port','time'], nemo_extracted.ssh.values.T),
                        ssh_obs  = (['port','time'], obs.ssh.values),
                        ntr_mod = (['port', 'time'], ntr_mod_all),
                        ntr_obs = (['port','time'], ntr_obs_all),
                        amp_mod = (['port','constituent'], a_mod),
                        amp_obs = (['port','constituent'], a_obs),
                        pha_mod = (['port','constituent'], g_mod),
                        pha_obs = (['port','constituent'], g_obs),
                        amp_err = (['port','constituent'], a_mod - a_obs),
                        pha_err = (['port','constituent'], compare_phase(g_mod, g_obs)),
                        std_obs = (['port'], std_obs),
                        std_mod = (['port'], std_mod),
                        std_err = (['port'], std_mod - std_obs),
                        ntr_corr = (['port'], ntr_corr),
                        ntr_mae  = (['port'], ntr_mae),
                        skew_mod = (['port', 'tide_num'], skew_mod_np),
                        skew_obs = (['port', 'tide_num'], skew_obs_np),
                        skew_err = (['port', 'tide_num'], skew_mod_np - skew_obs_np),
                        thresh_freq_ntr_mod  = (['port', 'threshold'], thresh_freq_ntr_mod),
                        thresh_freq_ntr_obs  = (['port', 'threshold'], thresh_freq_ntr_obs),
                        thresh_freq_skew_mod = (['port', 'threshold'], thresh_freq_skew_mod),
                        thresh_freq_skew_obs = (['port', 'threshold'], thresh_freq_skew_obs),
                        thresh_int_ntr_mod =(['port', 'threshold'], thresh_int_ntr_mod),
                        thresh_int_ntr_obs = (['port', 'threshold'], thresh_int_ntr_obs),
                        ntr_mod_clim_var     = (['port','clim_month'], ntr_clim_var.ntr_mod.values.T),
                        ntr_mod_clim_mean    = (['port','clim_month'], ntr_clim_mean.ntr_mod.values.T),
                        ntr_mod_monthly_var  = (['port','time_month'], ntr_monthly_var.ntr_mod.values.T),
                        ntr_mod_monthly_mean = (['port','time_month'], ntr_monthly_mean.ntr_mod.values.T),
                        ntr_mod_monthly_max  = (['port','time_month'], ntr_monthly_max.ntr_mod.values.T), 
                        ntr_obs_clim_var     = (['port','clim_month'], ntr_clim_var.ntr_obs.values.T),
                        ntr_obs_clim_mean    = (['port','clim_month'], ntr_clim_mean.ntr_obs.values.T),
                        ntr_obs_monthly_var  = (['port','time_month'], ntr_monthly_var.ntr_obs.values.T),
                        ntr_obs_monthly_mean = (['port','time_month'], ntr_monthly_mean.ntr_obs.values.T),
                        ntr_obs_monthly_max  = (['port','time_month'], ntr_monthly_max.ntr_obs.values.T))
                        )
        
        write_stats_to_file(stats, fn_out)