import numpy as np
import xarray as xr

fn_anal = "/Users/dbyrne/Projects/CO9_AMM15/data/TIDEANAL.EDITED"
fn_out = "/Users/dbyrne/Projects/CO9_AMM15/data/tideanal.edited.nc"

fid = open(fn_anal)

line_count=0
longitude = []
latitude = []
ana_length = []

for line in fid.readlines():

    letter = line[:2]
    port_name = line[2:20]
    latitude.append(float( line[19:29] ))
    longitude.append(float( line[29:39] ))
    ana_length.append(float( line[39:44] ))
    year = int( line[44:48] )

ds_out = xr.Dataset(coords = dict(
                        longitude = ('port', longitude),
                        latitude = ('port', latitude)),
                    data_vars = dict(
                        ana_length = ('port',ana_length)))
ds_out.to_netcdf(fn_out)

scatter_kwargs = {'zorder':100, 'vmin':0, 'vmax':365}
f,a = pu.geo_scatter(ds_out.longitude, ds_out.latitude, c=ds_out.ana_length, 
                     s=5, scatter_kwargs = scatter_kwargs)