import sys
sys.path.append('/Users/dbyrne/code/COAsT')
import xarray as xr
import pandas as pd
import glob
import numpy as np
from datetime import datetime
import scipy.interpolate as interpolate
import coast

fn_gesla = "/Users/dbyrne/data/gesla/*"
fn_out = "/Users/dbyrne/data/bodc/gesla_2004_2014.nc"

start_date = datetime(2004,1,1)
end_date = datetime(2014,12,31,23)
lonbounds = [-15, 15]
latbounds= [45, 65]

time = pd.date_range(start_date, end_date, freq='1H')

file_list = glob.glob(fn_gesla)

n_port = len(file_list)
n_time = len(time)

longitude = []
latitude = []
site_name = []
contributor = []
ssh = []
qc_flags = []

tg = coast.TIDEGAUGE()
file_count = 0

for ff in range(0,len(file_list)):
    file = file_list[ff]
    print(str(ff) + '_' + file)
    
    # Read header
    header = tg.read_gesla_header_v3(file)
    lon_tmp = header['longitude']
    lat_tmp = header['latitude']
    if lon_tmp>lonbounds[1] or lon_tmp<lonbounds[0] or lat_tmp>latbounds[1] or lat_tmp<latbounds[0]:
        continue
    
    data = tg.read_gesla_to_xarray_v3(file)
    
    longitude[ff].append( data['longitude'] )
    latitude[ff].append( data['latitude'] )
    ssh_tmp = np.array(r[' "Data value"'])
    qc_tmp = np.array( r['QC flag'] )
    time_tmp = pd.to_datetime(r['Date'])
    
    ssh_tmp[qc_tmp!='  '] = np.nan
    ds_tmp = xr.Dataset(coords = dict(
                            time = ('time',time_tmp)),
                        data_vars = dict(
                            ssh = ('time',ssh_tmp)))
    ds_int = ds_tmp.interp(time=time, method='nearest')
    ssh[ff] = ds_int.ssh
    file_count +=1
    
ds = xr.Dataset( coords = dict(
                     time = ('time', time),
                     longitude = ('port',longitude),
                     latitude = ('port', latitude)),
                 data_vars = dict(
                     ssh = (['port','time'], ssh)))
ds.to_netcdf(fn_out)
