#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:57:22 2021

@author: dbyrne
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fft
from scipy import interpolate
import coast
import xarray as xr
import datetime
import matplotlib.dates as mdates
import utide

def compute_fft(s, sampling_rate, n = None):
    '''Computes an FFT on signal s using numpy.fft.fft.
    
       Parameters:
        s (np.array): the signal
        sampling_rate (num): sampling rate
        n (integer): If n is smaller than the length of the input, the input is cropped. If n is 
            larger, the input is padded with zeros. If n is not given, the length of the input signal 
            is used (i.e., len(s))
    '''
    if n == None:
        n = len(s)
        
    s[np.isnan(s)] = np.nanmean(s)
        
    fft_result = scipy.fft.fft(s, n)
    num_freq_bins = len(fft_result)
    fft_freqs = scipy.fft.fftfreq(num_freq_bins, d = 1 / sampling_rate)
    half_freq_bins = num_freq_bins // 2

    fft_freqs = fft_freqs[:half_freq_bins]
    fft_result = fft_result[:half_freq_bins]
    
    fft_phases = np.angle(fft_result)
    fft_amplitudes = np.abs(fft_result)
    fft_amplitudes = 2 * fft_amplitudes / (len(s))
    
    fft_freqs = (fft_freqs/sampling_rate)*24
    
    return fft_freqs, fft_amplitudes, fft_phases

def plot_with_harmonics(txty = 10, const = ['m2','k1','m4','mf'], log=False):
    
    speeds = {'m2':28.974,'s2':30,'k1':15.041,'o1':13.942,'m4':57.968,
              'm8':115.9364,'mf':1.09803,'sa':0.0410686}

    values = np.array( [speeds[cc] for cc in const] ).astype(float)
    values = 24*values/360
    
    f,a = plt.subplots(1,1)
    
    for ii in range(0,len(values)):
        if log:
            a.semilogy([values[ii], values[ii]],[0,txty], color='k', linewidth=0.75,
                   linestyle='--')
        else:
            a.plot([values[ii], values[ii]],[0,txty], color='k', linewidth=0.75,
                   linestyle='--')
        a.text(values[ii]-0.03,txty,const[ii])
    
    return f,a

def get_nearest(array, values):
    ind = np.array( [np.argmin(np.abs(array-vv)) for vv in values] )
    return ind

speeds = {'m2':28.974,'s2':30,'k1':15.041,'o1':13.942,'m4':57.968,
              'm8':115.9364,'mf':1.09803,'sa':0.0410686}

const = np.array( list(speeds.keys()) )
values = np.array( [speeds[cc] for cc in const] ).astype(float)
values = 24*values/360

fn_tidegauge = '/Users/dbyrne/data/gesla/gladstone-p234-uk-bodc'
fn_model = '/Users/dbyrne/Projects/CO9_AMM15/p0_liverpool.nc'

# Read tidegauge data
tg = coast.TIDEGAUGE(fn_tidegauge, date_start = datetime.datetime(2011,1,1), date_end = datetime.datetime(2012,1,1))
tg_ssh = tg.dataset.sea_level.values
tg_ssh[tg.dataset.qc_flags>3] = np.nan
tg_ssh = tg_ssh - np.nanmean(tg_ssh)
tg_time = tg.dataset.time.values

# Basic QC
std = np.nanstd(tg_ssh)
ind_qc = np.abs(tg_ssh) > 2.2*std
tg_ssh[ind_qc] = np.nan

# Read model data
mod = xr.open_dataset(fn_model)
mod_time = mod.time.values
mod_ssh = mod.ssh.values
mod_ssh = mod_ssh - np.nanmean(mod_ssh)
mod_ssh[np.isnan(tg_ssh)] = np.nan

# Time series analysis tests
fs60 = 1/(60*60)
fs15 = 1/(15*60)

# FFT
f_tg, a_tg, g_tg = compute_fft(tg_ssh, fs60)
f_mod, a_mod, g_mod = compute_fft(mod_ssh, fs60)

i_tg = scipy.signal.find_peaks(a_tg)[0]
i_mod = scipy.signal.find_peaks(a_mod)[0]

f_tg_peaks = f_tg[i_tg]
a_tg_peaks = a_tg[i_tg]
f_mod_peaks = f_mod[i_mod]
a_mod_peaks = a_mod[i_mod]

nn_tg = get_nearest(f_tg_peaks, values)
nn_mod = get_nearest(f_mod_peaks, values)

f_tg_const = f_tg_peaks[nn_tg]
a_tg_const = a_tg_peaks[nn_tg]
f_mod_const = f_mod_peaks[nn_mod]
a_mod_const = a_mod_peaks[nn_mod]

# Cross Spectral Density
f_xy, pxy = scipy.signal.csd(tg_ssh, mod_ssh, fs60)

# Spectrogram
sg_tg = scipy.signal.spectrogram(tg_ssh)
sg_mod = scipy.signal.spectrogram(mod_ssh)

# Spectral Coherence
co = scipy.signal.coherence(tg_ssh, mod_ssh, fs60, nperseg=1024)

# Harmonic Analysis
hat_tg = mdates.date2num(tg_time)
hat_mod = mdates.date2num(mod_time)
uts_tg = utide.solve(hat_tg, tg_ssh, lat=53.45)
uts_mod = utide.solve(hat_mod, mod_ssh, lat=53.45)

tg_rec = utide.reconstruct(hat_tg, uts_tg).h
tg_rec[np.isnan(tg_ssh)] = np.nan

fr, ar, gr = compute_fft(tg_ssh - tg_rec, fs60)