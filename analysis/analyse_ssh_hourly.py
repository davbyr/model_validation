import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import utide as ut
import scipy.signal as signal
import os

class analyse_ssh_hourly():
    
    def __init__(self, fn_nemo_data, fn_nemo_domain, fn_obs, fn_out,
                         thresholds = np.arange(0,2,0.1),
                         constit_to_save = ['M2', 'S2', 'K1','O1'],
                         coast_dev_dir = None):
        
        if coast_dev_dir is not None:
            import sys
            sys.path.append(coast_dev_dir)
        
        import coast
        import coast.general_utils as gu
            
        nemo = self.read_nemo_ssh(fn_nemo_data, fn_nemo_domain)
        
        landmask = self.read_nemo_landmask_using_top_level(fn_nemo_domain)
        
        obs = self.read_obs_data(fn_obs)
        
        obs = self.subset_obs_by_lonlat(nemo, obs)
        
        nemo_extracted, obs = self.extract_obs_locations(nemo, obs, landmask)
        
        obs = self.align_timings(nemo_extracted, obs)
        
        # Define Dimension Sizes
        n_port = obs.dims['port']
        n_time = obs.dims['t_dim']
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
            mod_ssh = port_mod.values
            obs_ssh = port_obs.ssh.values
            shared_mask = np.logical_or(np.isnan(mod_ssh), np.isnan(obs_ssh))
            mod_ssh = np.ma.array(mod_ssh, mask = shared_mask)
            obs_ssh = np.ma.array(obs_ssh, mask = shared_mask)
            mod_time = port_mod.time.values
            obs_time = port_obs.time.values
            
            if len(np.where(~obs_ssh.mask)[0]) < 672:
                skew_mod.append([])
                skew_obs.append([])
                continue
            
            # Harmonic analysis datenums
            hat  = mdates.date2num(port_mod.time.values)
            
            # Do harmonic analysis using UTide
            uts_obs = ut.solve(hat, obs_ssh, lat=port_obs.latitude.values)
            uts_mod = ut.solve(hat, mod_ssh, lat=port_mod.latitude.values)
            
            # Reconstruct tidal signal 
            obs_tide = np.ma.array( ut.reconstruct(hat, uts_obs).h, mask=shared_mask)
            mod_tide = np.ma.array( ut.reconstruct(hat, uts_mod).h, mask=shared_mask)
            
            # Identify Peaks in tide and TWL 
            
            pk_ind_mod_tide,_ = signal.find_peaks(mod_tide, distance = 9)
            pk_ind_obs_tide,_ = signal.find_peaks(obs_tide, distance = 9)
            pk_ind_mod_ssh,_  = signal.find_peaks(mod_ssh, distance = 9)
            pk_ind_obs_ssh,_  = signal.find_peaks(obs_ssh, distance = 9)
            
            pk_time_mod_tide = pd.to_datetime( mod_time[pk_ind_mod_tide] )
            pk_time_obs_tide = pd.to_datetime( obs_time[pk_ind_obs_tide] )
            pk_time_mod_ssh  = pd.to_datetime( mod_time[pk_ind_mod_ssh] )
            pk_time_obs_ssh  = pd.to_datetime( obs_time[pk_ind_obs_ssh] )
            
            pk_mod_tide = mod_tide[pk_ind_mod_tide]
            pk_obs_tide = obs_tide[pk_ind_obs_tide] 
            pk_mod_ssh  = mod_ssh[pk_ind_mod_ssh]
            pk_obs_ssh  = obs_ssh[pk_ind_obs_ssh]
            
            # Define Skew Surges
            n_tide_mod = len(pk_mod_tide)
            n_tide_obs = len(pk_obs_tide)
            
            pk_mod_ssh_interp = np.zeros(n_tide_mod)
            pk_obs_ssh_interp = np.zeros(n_tide_obs)
            
            # Model Skew Surge
            for ii in range(0, n_tide_mod):
                time_diff = np.abs(pk_time_mod_tide[ii] - pk_time_mod_ssh)
                search_ind = np.where(time_diff < timedelta(hours=6))
                if len(search_ind[0]) > 0:
                    pk_mod_ssh_interp[ii] = np.nanmax(pk_mod_ssh[search_ind[0]])
                else:
                    pk_mod_ssh_interp[ii] = np.nan
                    
            # Observed Skew Surge
            pk_obs_ssh_interp = np.zeros(n_tide_obs)
            for ii in range(0, n_tide_obs):
                time_diff = np.abs(pk_time_obs_tide[ii] - pk_time_obs_ssh)
                search_ind = np.where(time_diff < timedelta(hours=6))
                if len(search_ind[0]) > 0:
                    pk_obs_ssh_interp[ii] = np.nanmax(pk_obs_ssh[search_ind])
                else:
                    pk_obs_ssh_interp[ii] = np.nan
                    
            skew_mod_tmp = pk_mod_ssh_interp - pk_mod_tide
            skew_obs_tmp = pk_obs_ssh_interp - pk_obs_tide
            
            ds_tmp = xr.Dataset(coords = dict(
                                    time = ('time',pk_time_mod_tide)),
                                data_vars = dict(
                                    ssh = ('time',skew_mod_tmp)))
            ds_int = ds_tmp.interp(time=pk_time_obs_tide, method='nearest')
            skew_mod_tmp = ds_int.ssh.values
        
            skew_mod.append(skew_mod_tmp)
            skew_obs.append(skew_obs_tmp)
            skew_err.append(skew_mod_tmp - skew_obs_tmp)
            
            # TWL: Basic stats
            std_obs[pp] = np.ma.std(obs_ssh)
            std_mod[pp] = np.ma.std(mod_ssh)
            std_err[pp] = np.ma.std(mod_ssh) - np.ma.std(obs_ssh)
            
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
            ntr_obs = obs_ssh - obs_tide
            ntr_mod = mod_ssh - mod_tide
            
            ntr_obs = signal.savgol_filter(ntr_obs,25,3)
            ntr_mod = signal.savgol_filter(ntr_mod,25,3)
            
            ntr_obs = np.ma.masked_invalid(ntr_obs)
            ntr_mod = np.ma.masked_invalid(ntr_mod)
            
            ntr_obs_all[pp] = ntr_obs
            ntr_mod_all[pp] = ntr_mod
            
            pk_ind_ntr_obs,_ = signal.find_peaks(ntr_obs, distance = 12)
            pk_ind_ntr_mod,_ = signal.find_peaks(ntr_mod, distance = 12)
            
            pk_time_ntr_obs = pd.to_datetime( obs_time[pk_ind_ntr_obs] )
            pk_time_ntr_mod = pd.to_datetime( mod_time[pk_ind_ntr_mod] )
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
                        time = ('time', obs.time.values),
                        constituent = ('constituent', constit_to_save),
                        threshold = ('threshold', thresholds)),
                   data_vars = dict(
                        nemo_ssh = (['port','time'], nemo_extracted.values.T),
                        obs_ssh  = (['port','time'], obs.ssh.values),
                        amp_mod = (['port','constituent'], a_mod),
                        amp_obs = (['port','constituent'], a_obs),
                        pha_mod = (['port','constituent'], g_mod),
                        pha_obs = (['port','constituent'], g_obs),
                        amp_err = (['port','constituent'], a_mod - a_obs),
                        pha_err = (['port','constituent'], self.compare_phase(g_mod, g_obs)),
                        std_obs = (['port'], std_obs),
                        std_mod = (['port'], std_mod),
                        std_err = (['port'], std_err),
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
                        ntr_mod_clim_var     = (['port','month'], ntr_clim_var.ntr_mod.values.T),
                        ntr_mod_clim_mean    = (['port','month'], ntr_clim_mean.ntr_mod.values.T),
                        ntr_mod_monthly_var  = (['port','month'], ntr_monthly_var.ntr_mod.values.T),
                        ntr_mod_monthly_mean = (['port','month'], ntr_monthly_mean.ntr_mod.values.T),
                        ntr_mod_monthly_max  = (['port','month'], ntr_monthly_max.ntr_mod.values.T), 
                        ntr_obs_clim_var     = (['port','month'], ntr_clim_var.ntr_obs.values.T),
                        ntr_obs_clim_mean    = (['port','month'], ntr_clim_mean.ntr_obs.values.T),
                        ntr_obs_monthly_var  = (['port','month'], ntr_monthly_var.ntr_obs.values.T),
                        ntr_obs_monthly_mean = (['port','month'], ntr_monthly_mean.ntr_obs.values.T),
                        ntr_obs_monthly_max  = (['port','month'], ntr_monthly_max.ntr_obs.values.T))
                        )
        
        self.write_stats_to_file(stats, fn_out)
        
    def write_stats_to_file(self, stats, fn_out):
        if os.path.exists(fn_out):
            os.remove(fn_out)
        stats.to_netcdf(fn_out)
        print("analyse_monthly_ssh: Statistics written to file")


    def read_nemo_ssh(self, fn_nemo_data, fn_nemo_domain):
        nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain, 
                                  multiple=True, chunks={'time_counter':168}).dataset
        return nemo['ssh']
    
    def read_nemo_landmask_using_top_level(self, fn_nemo_domain):
        dom = xr.open_dataset(fn_nemo_domain)
        landmask = np.array(dom.top_level.values.squeeze() == 0)
        dom.close()
        print("analyse_ssh_hourly: landmask defined")
        return landmask
    
    def read_obs_data(self, fn_obs):
        return xr.open_dataset(fn_obs)
    
    def subset_obs_by_lonlat(self, nemo, obs):
        lonmax = np.nanmax(nemo.longitude)
        lonmin = np.nanmin(nemo.longitude)
        latmax = np.nanmax(nemo.latitude)
        latmin = np.nanmin(nemo.latitude)
        ind = gu.subset_indices_lonlat_box(obs.longitude, obs.latitude, 
                                           lonmin, lonmax, latmin, latmax)
        obs = obs.isel(port=ind[0])
        print("analyse_ssh_hourly: obs data subsetted")
        return obs
    
    def extract_obs_locations(self, nemo, obs, landmask):
        # Extract model locations
        ind2D = gu.nearest_indices_2D(nemo.longitude, nemo.latitude, 
                                    obs.longitude, obs.latitude,
                                    mask = landmask)
        nemo_extracted = nemo.isel(x_dim = ind2D[0], y_dim = ind2D[1]).load()
        nemo_extracted = nemo_extracted.swap_dims({'dim_0':'port'})
        
        # Check interpolation distances
        max_dist = 5
        interp_dist = gu.calculate_haversine_distance(nemo_extracted.longitude, 
                                                      nemo_extracted.latitude, 
                                                      obs.longitude.values,
                                                      obs.latitude.values)
        keep_ind = interp_dist < max_dist
        nemo_extracted = nemo_extracted.isel(port=keep_ind)
        obs = obs.isel(port=keep_ind)
        print("analyse_ssh_hourly: obs location extracted from model data")
        return nemo_extracted, obs
    
    def align_timings(self, nemo_extracted, obs):
        obs = obs.interp(time = nemo_extracted.time, method = 'nearest')
        return obs
    
    def compare_phase(self, g1, g2):
        g1 = np.array(g1)
        g2 = np.array(g2)
        r = (g1-g2)%360 - 360
        r[r<-180] = r[r<-180] + 360
        r[r>180] = r[r>180] - 360
        return r