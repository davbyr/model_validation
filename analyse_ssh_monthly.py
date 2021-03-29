import sys
sys.path.append('/home/users/dbyrne/code/COAsT')
import coast
import coast.general_utils as gu
import coast.plot_util as pu
import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import matplotlib.pyplot as plt

class analyse_ssh_monthly():
    
    def __init__(self, fn_nemo_data, fn_nemo_domain, fn_psmsl_monthly, fn_out,
                 nemo_chunks = {}):
        
        nemo = self.read_nemo_data(fn_nemo_data, fn_nemo_domain, 
                                   chunks=nemo_chunks)
        landmask = self.read_nemo_landmask_using_top_level(fn_nemo_domain)
        psmsl = self.read_psmsl_data(fn_psmsl_monthly)
        psmsl = self.subset_psmsl_by_lonlat(nemo, psmsl)
        nemo_extracted, psmsl = self.extract_tg_locations(nemo, psmsl, landmask)
        nemo_extracted, psmsl = self.align_times(nemo_extracted, psmsl)
        nemo_extracted, psmsl = self.subtract_means(nemo_extracted, psmsl)
        stats = self.calculate_statistics(nemo_extracted, psmsl)
        self.write_stats_to_file(stats, fn_out)
        
        return
    
    def read_nemo_data(self, fn_nemo_data, fn_nemo_domain, chunks):
        nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain, 
                          multiple=True, chunks={}).dataset
        nemo = nemo['ssh']
        print("analyse_monthly_ssh: NEMO data read")
        return nemo
    
    def read_nemo_landmask_using_top_level(self, fn_nemo_domain):
        dom = xr.open_dataset(fn_nemo_domain)
        landmask = np.array(dom.top_level.values.squeeze() == 0)
        dom.close()
        print("analyse_monthly_ssh: landmask defined")
        return landmask
    
    def read_psmsl_data(self, fn_psmsl_monthly):
        # Sort out observation data
        psmsl = xr.open_dataset(fn_psmsl_monthly)
        psmsl['height'] = psmsl['height']/1000
        print("analyse_monthly_ssh: psmsl data read")
        return psmsl
    
    def subset_psmsl_by_lonlat(self, nemo, psmsl):
        lonmax = np.nanmax(nemo.longitude)
        lonmin = np.nanmin(nemo.longitude)
        latmax = np.nanmax(nemo.latitude)
        latmin = np.nanmin(nemo.latitude)
        ind = gu.subset_indices_lonlat_box(psmsl.longitude, psmsl.latitude, 
                                           lonmin, lonmax, latmin, latmax)
        psmsl = psmsl.isel(port=ind[0])
        print("analyse_monthly_ssh: PSMSL data subsetted")
        return psmsl
    
    def extract_tg_locations(self, nemo, psmsl, landmask):
        # Extract model locations
        ind2D = gu.nearest_indices_2D(nemo.longitude, nemo.latitude, 
                                    psmsl.longitude, psmsl.latitude,
                                    mask = landmask)
        nemo_extracted = nemo.isel(x_dim = ind2D[0], y_dim = ind2D[1]).load()
        nemo_extracted = nemo_extracted.swap_dims({'dim_0':'port'})
        
        # Check interpolation distances
        max_dist = 5
        interp_dist = gu.calculate_haversine_distance(nemo_extracted.longitude, 
                                                      nemo_extracted.latitude, 
                                                      psmsl.longitude.values,
                                                      psmsl.latitude.values)
        keep_ind = interp_dist < max_dist
        nemo_extracted = nemo_extracted.isel(port=keep_ind)
        psmsl = psmsl.isel(port=keep_ind)
        print("analyse_monthly_ssh: TG location extracted from model data")
        return nemo_extracted, psmsl
    
    def align_times(self, nemo_extracted, psmsl):
        # Select times
        nemo_month = pd.to_datetime(nemo_extracted.time.values).month
        nemo_year = pd.to_datetime(nemo_extracted.time.values).year
        psmsl_month = pd.to_datetime(psmsl.time.values).month
        psmsl_year = pd.to_datetime(psmsl.time.values).year
        
        nemo_extracted['time'] = ('t_dim', [datetime(nemo_year[ii], nemo_month[ii], 1) for ii in range(0, len(nemo_month)) ])
        psmsl['time'] = ('time', [datetime(psmsl_year[ii], psmsl_month[ii], 1) for ii in range(0, len(psmsl_month)) ] )
        print("analyse_monthly_ssh: Model and obs times aligned")
        psmsl = psmsl.interp(time=nemo_extracted.time, method='nearest')
        return nemo_extracted, psmsl
    
    def subtract_means(self, nemo_extracted, psmsl):
        psmsl['height'] = psmsl['height'] - psmsl['height'].mean(dim='t_dim')
        nemo_extracted = nemo_extracted - nemo_extracted.mean(dim='t_dim')
        print("analyse_monthly_ssh: Means during time period subtracted from data")
        return nemo_extracted, psmsl
    
    def calculate_statistics(self, nemo_extracted, psmsl):
        n_port = psmsl.dims['port']
        corr = np.zeros(n_port)*np.nan
        me = np.zeros(n_port)*np.nan
        mae = np.zeros(n_port)*np.nan
        std_mod = np.zeros(n_port)*np.nan
        std_obs = np.zeros(n_port)*np.nan
        std_err = np.zeros(n_port)*np.nan
        
        stats = xr.Dataset(coords = dict(
                                time = ('time', nemo_extracted.time.values),
                                longitude = ('port', psmsl.longitude.values),
                                latitude = ('port', psmsl.latitude.values)),
                           data_vars = dict(
                                nemo_extracted = (['port','time'], nemo_extracted.values.T),
                                psmsl_extracted = (['port','time'], psmsl.height.values)))
        
        for pp in range(0, psmsl.dims['port']):
            port_mod = nemo_extracted.isel(port=pp)
            port_obs = psmsl.isel(port=pp)
            
            if all(np.isnan(port_obs.height)):
                continue 
            
            # Masked arrays
            masked_obs = np.ma.masked_invalid(port_obs.height)
            masked_mod = np.ma.array(port_mod, mask=masked_obs.mask)
            
            # Standard Deviations
            std_mod[pp] = np.ma.std(masked_mod)
            std_obs[pp] = np.ma.std(masked_obs)
            std_err[pp] = std_mod[pp] - std_obs[pp]
            
            # Correlations
            c = np.ma.corrcoef(masked_obs, masked_mod)
            corr[pp] = c[1,0]
            
            # MAE
            errors = masked_mod - masked_obs
            me[pp] = np.ma.mean(errors)
            mae[pp] = np.ma.mean( np.ma.abs(errors) )
        
        stats['std_mod'] = ('port', std_mod)
        stats['std_obs'] = ('port', std_obs)
        stats['std_err'] = ('port', std_err)
        stats['corr'] = ('port', corr)
        stats['me'] = ('port', me)
        stats['mae'] = ('port', mae)
        
        print("analyse_monthly_ssh: Calculated statistics")
        return stats
    
    def write_stats_to_file(self, stats, fn_out):
        if os.path.exists(fn_out):
            os.remove(fn_out)
        stats.to_netcdf(fn_out)
        print("analyse_monthly_ssh: Statistics written to file")
        
class plot_monthly_ssh_stats_single_cfg():
    
    def __init__(self, fn_monthly_ssh_stats, dn_out, run_name, file_type='.png'):
    
        stats = xr.open_dataset(fn_monthly_ssh_stats)
        
        lonmax = np.max(stats.longitude)
        lonmin = np.min(stats.longitude)
        latmax = np.max(stats.latitude)
        latmin = np.min(stats.latitude)
        lonbounds = [lonmin-4, lonmax+4]
        latbounds = [latmin-4, latmax+4]
        
        # Plot correlations
        f,a = pu.create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats.longitude, stats.latitude, c=stats.corr, 
                        vmin=.75, vmax=1,
                  edgecolors='k', linewidths=.5, zorder=100)
        f.colorbar(sca)
        a.set_title('SSH correlations \n Monthly mean PSMSL tide gauge vs {0} \n Mean Corr = {1}'.format(run_name, np.nanmean(stats.corr)), fontsize=9)
        fn = "ssh_monthly_correlations_{0}{1}".format(run_name, file_type)
        f.savefig(os.path.join(dn_out, fn))
        plt.close('all')
        
        # Plot std_err
        f,a = pu.create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats.longitude, stats.latitude, c=stats.std_err, 
                        vmin=-.075, vmax=.075,
                  edgecolors='k', linewidths=.5, zorder=100, cmap='seismic')
        f.colorbar(sca)
        a.set_title('Standard Deviation Difference (m) \n Monthly mean {0} - PSMSL Tide Gauge \n Mean err = {1}'.format(run_name, np.nanmean(stats.std_err)), fontsize=9)
        fn = "ssh_monthly_stderr_{0}{1}".format(run_name, file_type)
        f.savefig(os.path.join(dn_out, fn))
        plt.close('all')
        
        # Plot me
        f,a = pu.create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats.longitude, stats.latitude, c=stats.me, 
                        vmin=-.05, vmax=.05,
                  edgecolors='k', linewidths=.5, zorder=100, cmap='seismic')
        f.colorbar(sca)
        a.set_title('Mean Error (m) \n Monthly mean {0} - PSMSL Tide Gauge \n Mean ME = {1}'.format(run_name, np.nanmean(stats.me)), fontsize=9)
        fn = "ssh_monthly_me_{0}{1}".format(run_name, file_type)
        f.savefig(os.path.join(dn_out, fn))
        plt.close('all')
        
        # Plot mae
        f,a = pu.create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats.longitude, stats.latitude, c=stats.mae, 
                        vmin=0, vmax=.05,
                  edgecolors='k', linewidths=.5, zorder=100)
        f.colorbar(sca)
        a.set_title('Mean Absolute Error (m) \n Monthly mean {0} - PSMSL Tide Gauge \n Mean MAE = {1}'.format(run_name, np.nanmean(stats.mae)), fontsize=9)
        fn = "ssh_monthly_mae_{0}{1}".format(run_name, file_type)
        f.savefig(os.path.join(dn_out, fn))
        plt.close('all')
        
        # Plot st.dev comparisons
        f,a = pu.scatter_with_fit(stats.std_mod, stats.std_obs)
        a.set_title('Monthly SSH standard deviation comparison (m) \n {0} vs {1}'.format(run_name,'PSMSL'), fontsize=9)
        a.set_xlabel("{0} SSH st. dev (m)".format(run_name))
        a.set_ylabel("PSMSL SSH st. dev (m)")
        fn = "ssh_monthly_stdev_scatter_{0}{1}".format(run_name, file_type)
        f.savefig(os.path.join(dn_out, fn))
        plt.close('all')
        
class plot_monthly_ssh_stats_multi_cfg():
    
    def __init__(self, fn_monthly_ssh_stats, dn_out, file_type, run_name):
    
        stats = [xr.open_dataset(fn) for fn in fn_monthly_ssh_stats]
        
        # Correlations
        
        # Std_err
        
        # MAE
        
        # ME
        
class plot_monthly_ssh_stats_comparison():
    
    def __init__(self, fn_monthly_ssh_stats0, fn_monthly_ssh_stats1, dn_out, 
                       run_names, file_type='.png'):
    
        stats0 = xr.open_dataset(fn_monthly_ssh_stats0)
        stats1 = xr.open_dataset(fn_monthly_ssh_stats1)
        
        lonmax = np.max(stats0.longitude)
        lonmin = np.min(stats0.longitude)
        latmax = np.max(stats0.latitude)
        latmin = np.min(stats0.latitude)
        lonbounds = [lonmin-4, lonmax+4]
        latbounds = [latmin-4, latmax+4]
        
        # Plot correlations
        f,a = pu.create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats0.longitude, stats0.latitude, 
                        c=stats0.corr - stats1.corr, 
                        vmin=-.03, vmax=.03,
                  edgecolors='k', linewidths=.5, zorder=100, cmap='PiYG')
        f.colorbar(sca)
        a.set_title("SSH correlation difference  {1} - {0} (m)\n Correlations between monthly mean model SSH and PSMSL".format(run_names[1], run_names[0]), fontsize=9)
        fn = "ssh_diff_corr_{0}_{1}{2}".format(run_names[0], run_names[1], file_type)
        f.savefig(os.path.join(dn_out, fn))
        plt.close('all')
        
        # Plot std_err
        f,a = pu.create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats0.longitude, stats0.latitude, 
                        c=np.abs(stats1.std_err)-np.abs(stats0.std_err), 
                        vmin=-.01, vmax=.01,
                  edgecolors='k', linewidths=.5, zorder=100, cmap='PiYG')
        f.colorbar(sca)
        a.set_title('Absolute St. Dev Error Difference {0} - {1} (m) \n Error between monthly mean model SSH and PSMSL'.format(run_names[1], run_names[0]), fontsize=9)
        fn = "ssh_diff_stderr_{0}_{1}{2}".format(run_names[0], run_names[1], file_type)
        f.savefig(os.path.join(dn_out, fn))
        plt.close('all')
        
        # Plot mae
        f,a = pu.create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats0.longitude, stats0.latitude, 
                        c=stats1.mae-stats0.mae, 
                        vmin=-.01, vmax=.01,
                  edgecolors='k', linewidths=.5, zorder=100, cmap='PiYG')
        f.colorbar(sca)
        a.set_title('MAE Difference {0} - {1} (m) \n Error between monthly mean model SSH and PSMSL'.format(run_names[1], run_names[0]), fontsize=9)
        fn = "ssh_diff_mae_{0}_{1}{2}".format(run_names[0], run_names[1], file_type)
        f.savefig(os.path.join(dn_out, fn))
        plt.close('all')
