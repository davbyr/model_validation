import numpy as np
import coast
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import utide

fn_tidegauge = '/Users/dbyrne/data/gesla/gladstone-p234-uk-bodc'

const_01 = ['M2']
const_02 = ['M2','S2','K2']
const_03 = ['M2','S2','K2','K1','O1','P1']
const_04 = ['M2','S2','K1','K1','O1','P1','M4','M6','M8']
const_05 = ['M2','S2','K1','K1','O1','P1','M4','M6','M8','MM','MF','SA','SSA']
const_auto = 'auto'
all_sets = [const_01, const_02, const_03, const_04, const_05, const_auto]

const_save = ['M2']
fn_out = '/Users/dbyrne/m2var.nc'

years = np.arange(1994, 2013)
base_year = datetime.datetime(1994,1,1)



# Read tidegauge data
tg = coast.TIDEGAUGE(fn_tidegauge, date_start=base_year).dataset
tg_ssh = tg.sea_level.values
tg_ssh[tg.qc_flags>3] = np.nan
tg_ssh = tg_ssh - np.nanmean(tg_ssh)

# Basic QC
std = np.nanstd(tg_ssh)
ind_qc = np.abs(tg_ssh) > 2.5*std
tg_ssh[ind_qc] = np.nan

tg['ssh'] = ('time', tg_ssh)

a_year = np.zeros((len(const_save), len(years), len(all_sets)))*np.nan
a_myear = np.zeros((len(const_save), len(years), len(all_sets)))*np.nan
g_year = np.zeros((len(const_save), len(years), len(all_sets)))*np.nan
g_myear = np.zeros((len(const_save), len(years), len(all_sets)))*np.nan

for yy in range(0,len(years)):
    print(yy)
    tg_year = tg.sel(time = slice(datetime.datetime(years[yy],1,1), datetime.datetime(years[yy]+1,1,1)))
        
    # Analyse year of data with different constituent sets
    for ss in range(0,len(all_sets)):
        cset = all_sets[ss]
        hat = mdates.date2num(tg_year.time)
        uts = utide.solve(hat, tg_year.ssh, lat=tg.latitude.values, constit=cset)
        a_dict = dict(zip(uts.name, uts.A))
        g_dict = dict(zip(uts.name, uts.g))
        for cc in range(0,len(const_save)):
            a_year[cc, yy, ss] = a_dict[const_save[cc]]
            g_year[cc, yy, ss] = g_dict[const_save[cc]]
            
    # Analyse different lengths of data
    if yy>0:
        tg_myear = tg.sel(time=slice(base_year, datetime.datetime(years[yy], 1, 1)))
        for ss in range(0,len(all_sets)):
            cset = all_sets[ss]
            hat = mdates.date2num(tg_myear.time)
            uts = utide.solve(hat, tg_myear.ssh, lat=tg.latitude.values, constit=cset)
            a_dict = dict(zip(uts.name, uts.A))
            g_dict = dict(zip(uts.name, uts.g))
            for cc in range(0,len(const_save)):
                a_myear[cc, yy-1, ss] = a_dict[const_save[cc]]
                g_myear[cc, yy-1, ss] = g_dict[const_save[cc]]

a_year = a_year.squeeze()
g_year = g_year.squeeze()
a_myear = a_myear.squeeze()
g_myear = g_myear.squeeze()
a_base = a_myear[-2,-1]
g_base = g_myear[-2,-1]
a_year_diff = (a_year - a_base).squeeze()       
g_year_diff = (g_year - g_base).squeeze() 
a_myear_diff = (a_myear - a_base).squeeze() 
g_myear_diff =  (g_myear - g_base).squeeze() 


out_ds = xr.Dataset(data_vars = dict(
                    a_year = (['year','cset'], a_year),
                    g_year = (['year','cset'], g_year),
                    a_myear = (['year','cset'], a_myear),
                    g_myear = (['year','cset'], g_myear),
                    a_year_diff = (['year','cset'], a_year_diff),
                    g_year_diff = (['year','cset'], g_year_diff),
                    a_myear_diff = (['year','cset'], a_myear_diff),
                    g_myear_diff = (['year','cset'], g_myear_diff)))

out_ds.to_netcdf(fn_out)