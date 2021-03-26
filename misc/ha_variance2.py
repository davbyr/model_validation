import numpy as np
import xarray as xr
import utide as ut
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import xarray.ufuncs as uf

def do_analysis(time, ssh, const, latitude):
    hat = mdates.date2num(time.values)
    uts_0 = ut.solve(hat, ssh.values, lat=latitude, constit=[const])
    uts_full = ut.solve(hat, ssh.values, lat=latitude, constit='auto')
    

    a_dict0 = dict( zip(uts_0.name, uts_0.A) )
    g_dict0 = dict( zip(uts_0.name, uts_0.g) )
    a_dict_full = dict( zip(uts_full.name, uts_full.A) )
    g_dict_full = dict( zip(uts_full.name, uts_full.g) )
    
    return a_dict0[const], g_dict0[const], a_dict_full[const], g_dict_full[const]

fn_tg = "/Users/dbyrne/data/bodc/tg_amm15.nc"
fn_out = '/Users/dbyrne/ha_var_m2.nc'

tg = xr.open_dataset(fn_tg)

const = "M2"

n_port = tg.dims['port']
n_sets = 2
n_const = len(const)

start_date = datetime(2003,12,31)
end_date = datetime(2014,12,31)

dates_1m = pd.date_range(start_date, end_date, freq='1M')
dates_3m = pd.date_range(start_date, end_date, freq='3M')
dates_6m = pd.date_range(start_date, end_date, freq='6M')
dates_12m = pd.date_range(start_date, end_date, freq='1Y')

n_1m = len(dates_1m)
n_3m = len(dates_3m)
n_6m = len(dates_6m)
n_12m = len(dates_12m)

a_1m = np.zeros((n_port, n_sets, n_1m-1)) * np.nan
g_1m = np.zeros((n_port, n_sets, n_1m-1)) * np.nan
a_3m = np.zeros((n_port, n_sets, n_3m-1)) * np.nan
g_3m = np.zeros((n_port, n_sets, n_3m-1)) * np.nan
a_6m = np.zeros((n_port, n_sets, n_6m-1)) * np.nan
g_6m = np.zeros((n_port, n_sets, n_6m-1)) * np.nan
a_12m = np.zeros((n_port, n_sets, n_12m-1)) * np.nan
g_12m = np.zeros((n_port, n_sets, n_12m-1)) * np.nan
a_10y = np.zeros((n_port, n_sets)) * np.nan
g_10y = np.zeros((n_port, n_sets)) * np.nan

a_1m_norm = np.zeros((n_port, n_sets, n_1m-1)) * np.nan
g_1m_norm = np.zeros((n_port, n_sets, n_1m-1)) * np.nan
a_3m_norm = np.zeros((n_port, n_sets, n_3m-1)) * np.nan
g_3m_norm = np.zeros((n_port, n_sets, n_3m-1)) * np.nan
a_6m_norm = np.zeros((n_port, n_sets, n_6m-1)) * np.nan
g_6m_norm = np.zeros((n_port, n_sets, n_6m-1)) * np.nan
a_12m_norm = np.zeros((n_port, n_sets, n_12m-1)) * np.nan
g_12m_norm = np.zeros((n_port, n_sets, n_12m-1)) * np.nan
a_10y_norm = np.zeros((n_port, n_sets)) * np.nan
g_10y_norm = np.zeros((n_port, n_sets)) * np.nan

for pp in range(0,n_port):
    tg_tmp = tg.isel(port=pp)
    
    a0,g0,af,gf = do_analysis(tg_tmp.time, tg_tmp.ssh, 
                              const, tg_tmp.latitude.values)
    a_10y[pp,0] = a0; g_10y[pp,0] = g0
    a_10y[pp,1] = af; g_10y[pp,1] = gf
    
    for dd in range(0,n_1m-1):
        
        tg_m = tg_tmp.sel(time=slice(dates_1m[dd], dates_1m[dd+1]))
        
        if len(np.where(uf.isnan(tg_m.ssh))[0]) > tg_m.dims['time']/5:
            continue
        a0,g0,af,gf = do_analysis(tg_m.time, tg_m.ssh, 
                              const, tg_m.latitude.values)
        a_1m[pp,0, dd] = a0; g_1m[pp,0,dd] = g0
        a_1m[pp,1, dd] = af; g_1m[pp,1,dd] = gf
        
    for dd in range(0,n_3m-1):
        
        tg_m = tg_tmp.sel(time=slice(dates_3m[dd], dates_3m[dd+1]))
        
        if len(np.where(uf.isnan(tg_m.ssh))[0]) > tg_m.dims['time']/5:
            continue
        a0,g0,af,gf = do_analysis(tg_m.time, tg_m.ssh, 
                              const, tg_m.latitude.values)
        a_3m[pp,0, dd] = a0; g_3m[pp,0,dd] = g0
        a_3m[pp,1, dd] = af; g_3m[pp,1,dd] = gf
        
    for dd in range(0,n_6m-1):
        
        tg_m = tg_tmp.sel(time=slice(dates_6m[dd], dates_6m[dd+1]))
        
        if len(np.where(uf.isnan(tg_m.ssh))[0]) > tg_m.dims['time']/5:
            continue
        a0,g0,af,gf = do_analysis(tg_m.time, tg_m.ssh, 
                              const, tg_m.latitude.values)
        a_6m[pp,0, dd] = a0; g_6m[pp,0,dd] = g0
        a_6m[pp,1, dd] = af; g_6m[pp,1,dd] = gf
        
    for dd in range(0,n_12m-1):
        
        tg_m = tg_tmp.sel(time=slice(dates_12m[dd], dates_12m[dd+1]))
        
        if len(np.where(uf.isnan(tg_m.ssh))[0]) > tg_m.dims['time']/5:
            continue
        a0,g0,af,gf = do_analysis(tg_m.time, tg_m.ssh, 
                              const, tg_m.latitude.values)
        a_12m[pp,0, dd] = a0; g_12m[pp,0,dd] = g0
        a_12m[pp,1, dd] = af; g_12m[pp,1,dd] = gf
        

for dd in range(0,n_1m-1):
    a_1m_norm[:,:,dd] = a_1m[:,:,dd]/a_10y
    g_1m_norm[:,:,dd] = g_1m[:,:,dd]-g_10y
for dd in range(0,n_3m-1):
    a_3m_norm[:,:,dd] = a_3m[:,:,dd]/a_10y
    g_3m_norm[:,:,dd] = g_3m[:,:,dd]-g_10y
for dd in range(0,n_6m-1):
    a_6m_norm[:,:,dd] = a_6m[:,:,dd]/a_10y
    g_6m_norm[:,:,dd] = g_6m[:,:,dd]-g_10y
for dd in range(0,n_12m-1):
    a_12m_norm[:,:,dd] = a_12m[:,:,dd]/a_10y
    g_12m_norm[:,:,dd] = g_12m[:,:,dd]-g_10y
    
a_1m_var = np.nanstd(a_1m, axis=2)
a_3m_var = np.nanstd(a_3m, axis=2)
a_6m_var = np.nanstd(a_6m, axis=2)
a_12m_var = np.nanstd(a_12m, axis=2)

a_1m_norm_var = np.nanstd(a_1m_norm, axis=2)
a_3m_norm_var = np.nanstd(a_3m_norm, axis=2)
a_6m_norm_var = np.nanstd(a_6m_norm, axis=2)
a_12m_norm_var = np.nanstd(a_12m_norm, axis=2)

g_1m_var = np.nanstd(g_1m, axis=2)
g_3m_var = np.nanstd(g_3m, axis=2)
g_6m_var = np.nanstd(g_6m, axis=2)
g_12m_var = np.nanstd(g_12m, axis=2)
        
ds = xr.Dataset(coords = dict(
                    longitude = ('port', tg.longitude),
                    latitude = ('port', tg.latitude),
                    time_1m = ('time_1m', dates_1m[:-1]),
                    time_3m = ('time_3m', dates_3m[:-1]),
                    time_6m = ('time_6m', dates_6m[:-1]),
                    time_12m = ('time_12m', dates_12m[:-1])),
                data_vars = dict(
                    a_1m = (['port','set','time_1m'], a_1m),
                    g_1m = (['port','set','time_1m'], g_1m),
                    a_3m = (['port','set','time_3m'], a_3m),
                    g_3m = (['port','set','time_3m'], g_3m),
                    a_6m = (['port','set','time_6m'], a_6m),
                    g_6m = (['port','set','time_6m'], g_6m),
                    a_12m = (['port','set','time_12m'], a_12m),
                    g_12m = (['port','set','time_12m'], g_12m),
                    a_10y = (['port','set'], a_10y),
                    g_10y = (['port','set'], g_10y),
                    a_1m_norm = (['port','set','time_1m'], a_1m_norm),
                    g_1m_norm = (['port','set','time_1m'], g_1m_norm),
                    a_3m_norm = (['port','set','time_3m'], a_3m_norm),
                    g_3m_norm = (['port','set','time_3m'], g_3m_norm),
                    a_6m_norm = (['port','set','time_6m'], a_6m_norm),
                    g_6m_norm = (['port','set','time_6m'], g_6m_norm),
                    a_12m_norm = (['port','set','time_12m'], a_12m_norm),
                    g_12m_norm = (['port','set','time_12m'], g_12m_norm),
                    a_10y_norm = (['port','set'], a_10y_norm),
                    g_10y_norm = (['port','set'], g_10y_norm),
                    a_1m_var = (['port','set'], a_1m_var),
                    a_3m_var = (['port','set'], a_3m_var),
                    a_6m_var = (['port','set'], a_6m_var),
                    a_12m_var = (['port','set'], a_12m_var),
                    a_1m_norm_var = (['port','set'], a_1m_norm_var),
                    a_3m_norm_var = (['port','set'], a_3m_norm_var),
                    a_6m_norm_var = (['port','set'], a_6m_norm_var),
                    a_12m_norm_var = (['port','set'], a_12m_norm_var),
                    g_1m_var = (['port','set'], g_1m_var),
                    g_3m_var = (['port','set'], g_3m_var),
                    g_6m_var = (['port','set'], g_6m_var),
                    g_12m_var = (['port','set'], g_12m_var)))

ds.to_netcdf(fn_out)
        
    
    
    
    


