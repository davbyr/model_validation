import sys
sys.path.append('/Users/dbyrne/code/COAsT')
import xarray as xr
import coast.plot_util as pu
import numpy as np
import matplotlib.pyplot as plt
import os.path
from scipy import interpolate

fn_out = '/Users/dbyrne/ha_var_m2.nc'
fn_ana = "/Users/dbyrne/Projects/CO9_AMM15/data/tideanal.edited.nc"
fn_fig = '/Users/dbyrne/Projects/CO9_AMM15/data/figs/ha_variance'

ha = xr.open_dataset(fn_out)
ana = xr.open_dataset(fn_ana)
const = 'M2'
file_type = '.png'

#%% Anal map plots

ind0 = ana.ana_length <= 30
ind1 = np.logical_and(ana.ana_length <= 90, ana.ana_length>30)
ind2 = np.logical_and(ana.ana_length <= 180, ana.ana_length>90)
ind3 = np.logical_and(ana.ana_length <= 365, ana.ana_length>180)
ind4 = ana.ana_length > 365

f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ana.longitude[ind0], ana.latitude[ind0], zorder=100, 
                s=2)
sca = a.scatter(ana.longitude[ind1], ana.latitude[ind1], zorder=70, 
                s=4)
sca = a.scatter(ana.longitude[ind2], ana.latitude[ind2], zorder=60, 
                s=6)
sca = a.scatter(ana.longitude[ind3], ana.latitude[ind3], zorder=50, 
                s=10)
sca = a.scatter(ana.longitude[ind4], ana.latitude[ind4], zorder=40, 
                s=10, marker='s', color='k')
f.legend(['l <= 30','30 < l <= 90','90 < l <= 180','180 < l <= 365','l > 365'],
         loc = 'lower right')
a.set_title('tideanal.edited locations and analysis length')
fn = "ana_lengths{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')


#%% Example time plots
pp = 2
ha_tmp = ha.isel(port = pp, set = 1)

# AMPLITUDE
f = plt.figure()
plt.plot([min(ha_tmp.time_1m.values), max(ha_tmp.time_1m.values)], [ha_tmp.a_10y, ha_tmp.a_10y], color='k', linestyle='--')
plt.plot(ha_tmp.time_1m, ha_tmp.a_1m)
plt.plot(ha_tmp.time_3m, ha_tmp.a_3m)
plt.plot(ha_tmp.time_6m, ha_tmp.a_6m)
plt.plot(ha_tmp.time_12m, ha_tmp.a_12m)
plt.grid()
plt.title('Amplitudes for different time periods | {0}N {1}E | {2}'.format(ha_tmp.latitude.values, ha_tmp.longitude.values, const), fontsize=10)
plt.ylabel('Amplitude (m)', fontsize=10)
plt.xlabel('Analysis Start Date', fontsize=10)
plt.legend(['10 Year','1 Month','3 Month','6 Month','1 Year'], loc = 'lower right')
fn = "eg_time_plot_amp{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# PHASE
f = plt.figure()
plt.plot([min(ha_tmp.time_1m.values), max(ha_tmp.time_1m.values)], [ha_tmp.g_10y, ha_tmp.g_10y], color='k', linestyle='--')
plt.plot(ha_tmp.time_1m, ha_tmp.g_1m)
plt.plot(ha_tmp.time_3m, ha_tmp.g_3m)
plt.plot(ha_tmp.time_6m, ha_tmp.g_6m)
plt.plot(ha_tmp.time_12m, ha_tmp.g_12m)
plt.grid()
plt.title('Amplitudes for different time periods | {0}N {1}E'.format(ha_tmp.latitude.values, ha_tmp.longitude.values), fontsize=10)
plt.ylabel('Phase (deg)', fontsize=10)
plt.xlabel('Analysis Start Date', fontsize=10)
plt.legend(['10 Year','1 Month','3 Month','6 Month','1 Year'], loc = 'lower right')
fn = "eg_time_plot_pha{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# Example Scatter
f = plt.figure()
plt.scatter(ha_tmp.a_12m-ha_tmp.a_10y, ha_tmp.g_12m-ha_tmp.g_10y, s=80)
plt.scatter(ha_tmp.a_6m-ha_tmp.a_10y, ha_tmp.g_6m-ha_tmp.g_10y, s=40)
plt.scatter(ha_tmp.a_3m-ha_tmp.a_10y, ha_tmp.g_3m-ha_tmp.g_10y, s=15)
plt.scatter(ha_tmp.a_1m-ha_tmp.a_10y, ha_tmp.g_1m-ha_tmp.g_10y, s=3, c='k')
plt.xlim(-0.05,0.05)
plt.ylim(-2,2)
plt.title('Difference from 10-year run | {0}N {1}E | {2}'.format(ha_tmp.latitude.values, ha_tmp.longitude.values, const), fontsize=10)
plt.xlabel('Amplitude Difference (m)')
plt.ylabel('Phase Difference (deg)')
plt.grid()
plt.legend(['1 Year','6 Month','3 Month','1 Month'], loc = 'lower right')
fn = "eg_scatter_plot{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

#%% All scatter
ha_tmp = ha.isel(set = 1)
plt.figure()
plt.scatter(1-ha_tmp.a_12m_norm, ha_tmp.g_12m-ha_tmp.g_10y, s=80)
plt.scatter(1-ha_tmp.a_6m_norm, ha_tmp.g_6m-ha_tmp.g_10y, s=40)
plt.scatter(1-ha_tmp.a_3m_norm, ha_tmp.g_3m-ha_tmp.g_10y, s=15)
plt.scatter(1-ha_tmp.a_1m_norm, ha_tmp.g_1m-ha_tmp.g_10y, s=3, c='k', alpha=0.2)
plt.xlim(-0.2,0.2)
plt.ylim(-25,25)
plt.xlabel('Proportional Amplitude Difference (m)')
plt.ylabel('Phase Difference (deg)')
plt.grid()
plt.legend(['1 Year','6 Month','3 Month','1 Month'], loc = 'lower right')
plt.title('Difference from 10-year run | All locations', fontsize=10)
fn = "all_scatter{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

#%% Location Variance - Absolute

# 1 Month absolute - Amplitude
f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ha.longitude, ha.latitude, c = ha.a_1m_var[:,1], zorder=100,
                vmin = 0, vmax=0.06)
f.colorbar(sca, extend='max')
a.set_title('Amplitude St. Dev. (1-month analyses)', fontsize=10)
fn = "map_amp_var_1m{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# 3 Month absolute - Amplitude
f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ha.longitude, ha.latitude, c = ha.a_3m_var[:,1], zorder=100,
                vmin = 0, vmax=0.06)
f.colorbar(sca, extend='max')
a.set_title('Amplitude St. Dev. (1-month analyses)', fontsize=10)
fn = "map_amp_var_3m{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# 6 Month absolute - Amplitude
f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ha.longitude, ha.latitude, c = ha.a_6m_var[:,1], zorder=100,
                vmin = 0, vmax=0.06)
f.colorbar(sca, extend='max')
a.set_title('Amplitude St. Dev. (1-month analyses)', fontsize=10)
fn = "map_amp_var_6m{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# 12 Month absolute - Amplitude
f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ha.longitude, ha.latitude, c = ha.a_12m_var[:,1], zorder=100,
                vmin = 0, vmax=0.06)
f.colorbar(sca)
a.set_title('Amplitude St. Dev. (1-month analyses)', fontsize=10)
fn = "map_amp_var_12m{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')


#%% Location Variance - Proportional
# 1 Month Proportional
f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ha.longitude, ha.latitude, c = ha.a_1m_norm_var[:,1], zorder=100,
                vmin = 0, vmax=0.03)
f.colorbar(sca)
a.set_title('Proportional Amplitude St. Dev. (1-month analyses)', fontsize=10)
fn = "map_amp_pvar_1m{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# 3 Month Proportional
f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ha.longitude, ha.latitude, c = ha.a_3m_norm_var[:,1], zorder=100,
                vmin = 0, vmax=0.03)
f.colorbar(sca)
a.set_title('Proportional Amplitude St. Dev. (3-month analyses)', fontsize=10)
fn = "map_amp_pvar_3m{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# 6 Month Proportional
f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ha.longitude, ha.latitude, c = ha.a_6m_norm_var[:,1], zorder=100,
                vmin = 0, vmax=0.03)
f.colorbar(sca)
a.set_title('Proportional Amplitude St. Dev. (6-month analyses)', fontsize=10)
fn = "map_amp_pvar_6m{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# 12 Month Proportional
f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ha.longitude, ha.latitude, c = ha.a_12m_norm_var[:,1], zorder=100,
                vmin = 0, vmax=0.03)
f.colorbar(sca)
a.set_title('Proportional Amplitude St. Dev. (12-month analyses)', fontsize=10)
fn = "map_amp_pvar_12m{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# %% Combined Variance plot - AMPLITUDE

mn1 = np.nanmean(ha.a_1m_norm_var[:,1]); mn3 = np.nanmean(ha.a_3m_norm_var[:,1])
mn6 = np.nanmean(ha.a_6m_norm_var[:,1]); mn12 = np.nanmean(ha.a_12m_norm_var[:,1])

xlab = np.array([1,3,6,12])
plt.figure()
plt.plot(xlab, [mn1, mn3, mn6, mn12], marker='o')
plt.title('Mean Proportional Amplitude Standard Deviation (m)', fontsize=10)
plt.xlabel('Analysis Length (Months)')
plt.ylabel('Proportional Amplitude St. Dev.')
plt.grid()
fn = "var_plot_amp{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')

# %% Interpolated Variance - AMPLITUDE

f = interpolate.interp1d(xlab*30, [mn1, mn3, mn6, mn12])
xnew = ana.ana_length.values
xnew[xnew<30] = 30
xnew[xnew>359] = 359
new = f(xnew)

f,a = pu.create_geo_axes([-15,15],[45,65])
sca = a.scatter(ana.longitude, ana.latitude, c=new, zorder=100, s=5)
f.colorbar(sca)
a.set_title('tideanal.edited M2 proportional error estimates')
fn = "map_amp_var_interpolated{0}".format(file_type)
f.savefig(os.path.join(fn_fig, fn))
plt.close('all')