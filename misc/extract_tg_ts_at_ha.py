import sys
sys.path.append('/home/users/dbyrne/code/COAsT/')
import coast
import coast.general_utils as gu
import xarray as xr
import numpy as np

fn_nemo_data = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/outputs/hourly/*.nc"
fn_nemo_domain = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/inputs/CO7_EXACT_CFG_FILE.nc"
fn_tideanal = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/obs/tideanal.edited.nc"
fn_out = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/analysis/tideanal_extracted.nc"

nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain, multiple=True, chunks='auto').dataset
tana = xr.open_dataset(fn_tideanal)
mask = nemo.bottom_level==0

ind2D = gu.nearest_indices_2D(nemo.longitude, nemo.latitude, 
                              tana.longitude, tana.latitude, mask)

nemo = nemo.isel(x_dim = ind2D[0], y_dim = ind2D[1])
nemo.compute()
nemo = nemo.swap_dims({'dim_0':'port'})
nemo = nemo.swap_dims({'t_dim':'time'})

nemo.to_netcdf(fn_out)