import sys
sys.path.append('/home/users/dbyrne/code/COAsT/')
import coast
import coast.general_utils as gu
import xarray as xr
import numpy as np

fn_nemo_data = "/Users/dbyrne/Projects/CO9_AMM15/data/nemo/2004*"
fn_nemo_domain = "/Users/dbyrne/Projects/CO9_AMM15/data/nemo/CO7_EXACT_CFG_FILE.nc"
fn_tideanal = "/Users/dbyrne/Projects/CO9_AMM15/data/tideanal.edited.nc"
fn_out = "/Users/dbyrne/Projects/CO9_AMM15/data/test.nc"

nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain, multiple=True, chunks={'time_counter':100}).dataset['ssh']
tana = xr.open_dataset(fn_tideanal)

ind2D = gu.nearest_indices_2D(nemo.longitude, nemo.latitude, 
                              tana.longitude, tana.latitude)

nemo = nemo.isel(x_dim = ind2D[0], y_dim = ind2D[1])
nemo = nemo.swap_dims({'dim_0':'port'})
nemo = nemo.swap_dims({'t_dim':'time'})

nemo.to_netcdf(fn_out)