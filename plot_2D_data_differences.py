#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 23:46:58 2021

@author: dbyrne
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

fn_data0 = "/Users/dbyrne/Projects/CO9_AMM15/ostia_seasonal_mean.nc"
fn_data1 = "/Users/dbyrne/Projects/CO9_AMM15/nemo_clim_regridded_to_ostia.nc"

ostia = xr.open_dataset(fn_clim0, chunks={})
nemo = xr.open_dataset(fn_clim1, chunks={})






# Winter
plt.pcolormesh(ostia.lon, ostia.lat, nemo.temperature[0] - ostia.temperature[0] )
plt.colorbar()
plt.clim(-1.5,1.5)
plt.set_cmap('seismic')
plt.grid(linewidth=0.5)
plt.title('SST difference (CO9p0 - OSTIA) \n DJF | $Mean = 0.056$ | degC')
plt.savefig("/Users/dbyrne/Projects/CO9_AMM15/clim_ostia_p0_djf.png")
plt.close('all')

# Winter
plt.pcolormesh(ostia.lon, ostia.lat, nemo.temperature[1] - ostia.temperature[1] )
plt.colorbar()
plt.clim(-1.5,1.5)
plt.set_cmap('seismic')
plt.grid(linewidth=0.5)
plt.title('SST difference (CO9p0 - OSTIA) \n JJA | $Mean = -0.019$ | degC')
plt.savefig("/Users/dbyrne/Projects/CO9_AMM15/clim_ostia_p0_jja.png")
plt.close('all')

# Winter
plt.pcolormesh(ostia.lon, ostia.lat, nemo.temperature[2] - ostia.temperature[2] )
plt.colorbar()
plt.clim(-1.5,1.5)
plt.set_cmap('seismic')
plt.grid(linewidth=0.5)
plt.title('SST difference (CO9p0 - OSTIA) \n MAM | $Mean = 0.053$ | degC')
plt.savefig("/Users/dbyrne/Projects/CO9_AMM15/clim_ostia_p0_mam.png")
plt.close('all')

# Winter
plt.pcolormesh(ostia.lon, ostia.lat, nemo.temperature[3] - ostia.temperature[3] )
plt.colorbar()
plt.clim(-1.5,1.5)
plt.set_cmap('seismic')
plt.grid(linewidth=0.5)
plt.title('SST difference (CO9p0 - OSTIA) \n SON | $Mean = -0.015$ | degC')
plt.savefig("/Users/dbyrne/Projects/CO9_AMM15/clim_ostia_p0_son.png")
plt.close('all')
