import xarray as xr
import pandas as pd
import glob
import numpy as np
from datetime import datetime
import scipy.interpolate as interpolate

fn_bodc = "/Users/dbyrne/data/bodc/*.csv"
fn_uhslc = "/Users/dbyrne/data/uhslc/*.nc"
fn_out = "/Users/dbyrne/data/bodc/tg_amm15.nc"
start_date = datetime(2004,1,1)
end_date = datetime(2014,12,31,23)
time = pd.date_range(start_date, end_date, freq='1H')

bodc_list = glob.glob(fn_bodc)
uhslc_list = glob.glob(fn_uhslc)
n_port = len(bodc_list) + len(uhslc_list)
n_time = len(time)

longitude = np.zeros(n_port)*np.nan
latitude = np.zeros(n_port)*np.nan
site_name = []
ssh = np.zeros((n_port, n_time))*np.nan

file_count = 0

for ff in range(0,len(bodc_list)):
    file = bodc_list[ff]
    print(file)
    r = pd.read_csv(file)
    longitude[ff] = r['Longitude'][0]
    latitude[ff] = r['Latitude'][0]
    ssh_tmp = np.array(r[' "Data value"'])
    qc_tmp = np.array( r['QC flag'] )
    time_tmp = pd.to_datetime(r['Date'])
    site_name.append(r['Site Name'][0])
    
    ssh_tmp[qc_tmp!='  '] = np.nan
    ds_tmp = xr.Dataset(coords = dict(
                            time = ('time',time_tmp)),
                        data_vars = dict(
                            ssh = ('time',ssh_tmp)))
    ds_int = ds_tmp.interp(time=time, method='nearest')
    ssh[ff] = ds_int.ssh
    file_count +=1
    
for ff in range(0, len(uhslc_list)):
    file = uhslc_list[ff]
    print(file)
    x = xr.open_dataset(file)
    site_name.append('N/A')
    
    longitude[file_count] = x.lon.values
    latitude[file_count] = x.lat.values
    
    time_tmp = x['time'].sel(time=slice(start_date, end_date)).values.squeeze()
    ssh_tmp = x['sea_level'].sel(time=slice(start_date, end_date)).values.squeeze()
    ssh_tmp = ssh_tmp/1000
    ds_tmp = xr.Dataset(coords = dict(
                            time = ('time',time_tmp)),
                        data_vars = dict(
                            ssh = ('time',ssh_tmp)))
    ds_int = ds_tmp.interp(time=time, method='nearest')
    ssh[file_count] = ds_int.ssh
    file_count += 1
    
ds = xr.Dataset( coords = dict(
                     time = ('time', time),
                     longitude = ('port',longitude),
                     latitude = ('port', latitude)),
                 data_vars = dict(
                     ssh = (['port','time'], ssh)))
ds.to_netcdf(fn_out)
