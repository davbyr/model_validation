"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.0 (20-01-2021)

This is a modular script for validating modelled elevation tidal harmonics against
point source tidal harmonics such as tide gauges. By default, the script is
set up to read NEMO tidal harmonics and compare them to MDP netcdf harmonics
(this is a restricted file so you may need to use your own). A nearest neighbour
interpolation is done to extract the nearest ocean model points to the point
observations (providing a landmask is supplied).

If you are using NEMO harmonics, this script may work for you. 
The modular nature of this script makes it easy to modify for different data.
The Python functions read_harmonics_obs() and read_harmonics_model() can be 
modified, or replaced as needed. So long as the output to these functions
is as specified in the function docstrings, then the rest of the validation
script will continue to work. In short, for different data, you only need to
modify these two Python functions for the script to work.

This script makes heavy use of xarray and also uses COAsT for some generic 
functions. It calculates basic statistics and plots them on some map figures.
These are then saved to a specified directory. The statistics data is also
saved to a new netCDF file, again in a specified directory.

See the main() method for the order of functions. At the top of the script are
the global variables which can be set. If you modify any functions, you may
have to change this variable list. By default, these variables are:
    
    fn_nemo_harmonics (str) :: Full path to NEMO harmonics file
    fn_nemo_domain (str)    :: Full path to NEMO domain file
    fn_obs_harmonics (str)  :: Full path to observed harmoncis
    dn_output (str)         :: Full path to output directory for figures and data
    constituents (list)     :: List of str for constituent names
    run_name (str)          :: String to use at beginning of file names and 
                               in figure titles
                               
*NOTE: If there are multiple data sources to read in (model or obs), this should
be done entirely within the read_harmonics_obs() or read_harmonics_model() 
routines. Data source should be combined into one xarray dataset and returned
to the main script.
"""

 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # GLOBAL VARIABLES
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
# Input paths and Filenames
fn_nemo_harmonics = "/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/subset2.nc"
fn_nemo_domain    = "/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/subset2_domain.nc"
fn_obs_harmonics = "/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/MDP_harmonics.nc"

# Output directory
dn_output = "/Users/Dave/Documents/Projects/CO9_AMM15/validation/plots"

# Constituents to analyse
constituents=['M2','S2', 'bl2']

# Name of run or configuration -- for file and figure naming.
run_name = "CO9_AMM15"





 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 #SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#




import coast
import coast.general_utils as coastgu
import coast.plot_util as coastpu
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path

def main():


    print(' *Tidal validation starting.* ')
    # ----------------------------------------------------
    # 1. Read observations. See comment in function for data format.
    obs_harmonics = read_harmonics_obs(fn_obs_harmonics, constituents)
    print(' 1. Observed harmonics read from file.')
    
    # ----------------------------------------------------
    # 2. Read model harmonics. See comment in function for data format
    model_harmonics = read_harmonics_model(fn_nemo_harmonics, fn_nemo_domain,
                                          constituents)
    print(' 2. NEMO harmonics read from file.')
    
    # ----------------------------------------------------
    # 3. Use only observations that are within model domain (Replace with COAsT
    # routine one day).
    lonmax = np.nanmax(model_harmonics.longitude)
    lonmin = np.nanmin(model_harmonics.longitude)
    latmax = np.nanmax(model_harmonics.latitude)
    latmin = np.nanmin(model_harmonics.latitude)
    ind = coast.general_utils.subset_indices_lonlat_box(obs_harmonics.longitude, 
                                                        obs_harmonics.latitude,
                                                        lonmin, lonmax, 
                                                        latmin, latmax)
    obs_harmonics = obs_harmonics.isel(port = ind[0])
    print(' 3. Observations subsetted to model domain max and min.')
    
    # ----------------------------------------------------
    # 4. Extract observation points from model (nearest model grid point). 
    # If mask is known, then nearest OCEAN point.
    ind2D = coast.general_utils.nearest_indices_2D(model_harmonics.longitude,
                                                   model_harmonics.latitude,
                                                   obs_harmonics.longitude,
                                                   obs_harmonics.latitude,
                                                   model_harmonics.landmask)
    model_harmonics = model_harmonics.isel(x_dim=ind2D[0], y_dim=ind2D[1])
    model_harmonics = model_harmonics.rename({'dim_0':'port'})
    print(' 4. Nearest model ocean points extracted.')
    
    # ----------------------------------------------------
    # 5. Calculate statistics
    stats = calculate_statistics(model_harmonics, obs_harmonics)
    print(' 5. Data compared, stats calculated.')
    
    # ----------------------------------------------------
    # 6. Plot and save errors on maps.
    plot_amplitude_errors_on_map(stats, dn_output)
    plot_abs_amplitude_errors_on_map(stats, dn_output)
    plot_phase_errors_on_map(stats, dn_output)
    plot_proportional_amplitude_errors_on_map(stats, dn_output)
    print(' 6. Plots saved to: ' + dn_output)
    
    # ----------------------------------------------------
    # 7. Write stats to new netcdf file
    print(' 7. Stats netcdf file saved to: ' + dn_output)
    
    # ----------------------------------------------------
    # Close files to be good and well behaved
    
    model_harmonics.close()
    obs_harmonics.close()
    print(' 8. Input files close. ' )
    print(' *Tidal validation complete.* ')
    

def read_harmonics_obs(fn_obs_harmonics, constituents):
    '''
    For reading and formatting observation data. Modify or replace as necessary.
    If output from this fucntion is correct then the validation script will
    continue to work. Data output should be in the form of a single xarray
    Dataset object. This Dataset must have the minimum form and variable names:
    
    Dimensions:           (constituent: 2, port: 1104)
    Coordinates:
        longitude         (port) float64 ...
        latitude          (port) float64 ...
        constituent_name  (constituent) object 'M2' 'S2'
    Data variables:
        amplitude         (constituent, port) float64 ...
        phase             (constituent, port) float64 ...
    '''

    # Open netCDF file and read data
    harmonics = xr.open_dataset(fn_obs_harmonics)
    
    # Select only the specified constituents and disregard the rest
    obs_constituents = harmonics.constituent_name
    ind = [np.where( obs_constituents == ss) \
                    for ss in constituents if ss in obs_constituents]
    ind = np.array(ind).T.squeeze()
    harmonics = harmonics.isel(constituent = ind)
    return harmonics

def read_harmonics_model(fn_nemo_harmonics, fn_nemo_domain, constituents):
    '''
    For reading and formatting model data. Modify or replace as necessary.
    If output from this fucntion is correct then the validation script will
    continue to work. Data output should be in the form of a single xarray
    Dataset object. This Dataset must have the minimum form and variable names:
    
    Dimensions:           (constituent: 2, x_dim: 369, y_dim: 297)
    Coordinates:
        latitude          (y_dim, x_dim) float32 ...
        longitude         (y_dim, x_dim) float32 ...
    Data variables:
        amplitude         (constituent, y_dim, x_dim) float64 0.7484 0.7524 ... 0.0
        phase             (constituent, y_dim, x_dim) float64 -1.8 -1.8 ... -0.0
        constituent_name  (constituent) <U3 'M2' 'S2'
        landmask          (y_dim, x_dim) bool False False False ... True True True
        
    Here, x_dim and y_dim are the longitude and latitude axes respectively. The
    default script determines the landmask from bathymetry. If not available,
    set this variable to all False (tell it there is no land -- a Lie!!).
    '''
    # Read model data into xarray dataset: both data and domain
    harmonics = xr.open_dataset(fn_nemo_harmonics)
    domain = xr.open_dataset(fn_nemo_domain)
    
    # Rename dimensions and coordinates -> CHECK THIS FOR YOUR DATA
    harmonics = harmonics.rename({'nav_lat_grid_T':'latitude',
                                  "nav_lon_grid_T":'longitude',
                                  "x_grid_T":"x_dim",
                                  "y_grid_T":"y_dim"})
    
    # Select only the specified constituents. NEMO model harmonics names are
    # things like "M2x" and "M2y". Ignore current harmonics. Start by constructing
    # the possible variable names
    names_x = np.array([cc + "x" for cc in constituents])
    names_y = np.array([cc + "y" for cc in constituents])
    constituents = np.array(constituents)
    
    # Compare against names in file
    var_keys = np.array(list(harmonics.keys()))
    indices = [np.where( names_x == ss) for ss in names_x if ss in var_keys]
    indices = np.array(indices).T.squeeze()
    
    # Index the possible names to match file names
    names_x = names_x[indices]
    names_y = names_y[indices]
    constituents = constituents[indices]
    
    # Create list of all variable names
    names = np.concatenate((names_x, names_y))
    
    # Extract only the variables that we want
    harmonics = harmonics[names]
    
    # Create empty amplitude and phase arrays
    n_constituents = len(constituents)
    n_x = harmonics.latitude.shape[0]
    n_y = harmonics.latitude.shape[1]
    amplitude = np.zeros((n_constituents, n_x, n_y))
    phase = np.zeros((n_constituents, n_x, n_y))
    
    # Convert to amplitude and phase (complex space -> polar)
    for ii in range(0,len(names_x)):
        var_x = harmonics[names_x[ii]]
        var_y = harmonics[names_y[ii]]
        amplitude[ii,:,:] = np.sqrt(var_x**2 + var_y**2)
        phase[ii,:,:] = np.arctan2(var_y, var_x)
        harmonics = harmonics.drop_vars(names_x[ii])
        harmonics = harmonics.drop_vars(names_y[ii])
        
    # Save new variables into xarray dataset
    harmonics['amplitude'] = (('constituent', 'y_dim', 'x_dim'),amplitude)
    harmonics['phase'] = (('constituent', 'y_dim', 'x_dim'),phase)
    harmonics['constituent_name'] = (('constituent'), constituents)
        
    # Define landmask as being everywhere that the depth is 0 (or "shallower")
    landmask = domain.bathy_metry<= 0
    landmask = landmask.squeeze()
    landmask = landmask.rename({"x":"x_dim", "y":"y_dim"})
    harmonics['landmask'] = (landmask)
    
    return harmonics

def calculate_statistics(model_harmonics, obs_harmonics):
    '''
    Calculates statistics for plottin and writing to file.
    Statistics are placed into a new xarray Dataset object.

    '''
    # Calculate statistics
    error_a = model_harmonics.amplitude - obs_harmonics.amplitude
    error_g = coastgu.compare_angles(model_harmonics.phase, obs_harmonics.phase)
    abs_error_a = np.abs(error_a)
    p_error_a = error_a.values / obs_harmonics.amplitude.values
    #mae_a = np.nanmean( abs_error_a.values )
    #mae_g = np.nanmean( error_g.values )
    
    # Place into stats xarray
    stats = xr.Dataset(data_vars = dict(
                         error_a = (["constituent","port"], error_a),
                         error_g = (["constituent","port"], error_g),
                         abs_error_a = (["constituent","port"], abs_error_a),
                         prop_error_a = (["constituent","port"], 
                                                 p_error_a),
                         #mae_a = (['constituent'], mae_a), 
                         #mae_g = (['constituent'], mae_g)
                     ),
                     coords = dict(
                         longitude=(["port"], obs_harmonics.longitude),
                         latitude=(["port"], obs_harmonics.latitude),
                         constituent_name=(["constituent"], obs_harmonics.constituent_name),
                         doodson_index=(["constituent"], obs_harmonics.doodson_index)
                     ),
                     attrs = dict(description='stats'))
    return stats

def plot_amplitudes_map(model_harmonics, obs_harmonics, dn_output):
    return

def plot_phases_map(model_harmonics, obs_harmonics, dn_output):
    return

def plot_amplitude_errors_on_map(stats, dn_output):
    ''' Plots proportional amplitude errors on a geographical map using 
    COAsT and cartopy. See plot_amplitude_errors_on_map() for more info'''
    
    for cc in range(0,len(stats.constituent)):
        stats_tmp = stats.isel(constituent=cc)
        cmax = np.nanmean(stats_tmp.error_a) + 3*np.nanstd(stats_tmp.error_a)
        scatter_kwargs = {"cmap":"coolwarm", "marker":"o", "ec":[0,0,0], 
                          'linewidth':0.5, "vmin":-cmax, "vmax":cmax}
        const_str = str(stats_tmp.constituent_name.values)
        title = 'Amplitude Error (m) | ' + run_name + ' | ' + const_str
        sca = coastpu.geo_scatter(stats.longitude, stats.latitude, 
                    c = stats_tmp.error_a, s=15,
                    scatter_kwargs=scatter_kwargs, title = title)
        fn_save = run_name + "_amplitude_error_"+const_str + '.eps'
        print("  >>>>>  Saving: "+ fn_save)
        plt.savefig(os.path.join(dn_output, fn_save))
        plt.close()
    return

def plot_phase_errors_on_map(stats, dn_output):
    ''' Plots proportional amplitude errors on a geographical map using 
    COAsT and cartopy. See plot_amplitude_errors_on_map() for more info'''
    
    for cc in range(0,len(stats.constituent)):
        stats_tmp = stats.isel(constituent=cc)
        cmax = np.nanmean(stats_tmp.error_g) + 3*np.nanstd(stats_tmp.error_g)
        cmin = np.nanmean(stats_tmp.error_g) - 3*np.nanstd(stats_tmp.error_g)
        cmin = np.max([cmin,0])
        scatter_kwargs = {"cmap":"Reds", "marker":"o", "ec":[0,0,0], 
                          'linewidth':0.5, "vmin":cmin, "vmax":cmax}
        const_str = str(stats_tmp.constituent_name.values)
        title = 'Phase Error (deg) | ' + run_name + ' | ' + const_str
        sca = coastpu.geo_scatter(stats.longitude, stats.latitude, 
                    c = stats_tmp.error_g, s=15,
                    scatter_kwargs=scatter_kwargs, title = title)
        fn_save = run_name + "_phase_error_"+const_str + '.eps'
        print("  >>>>>  Saving: "+ fn_save)
        plt.savefig(os.path.join(dn_output, fn_save))
        plt.close()
    return

def plot_abs_amplitude_errors_on_map(stats, dn_output):
    ''' Plots proportional amplitude errors on a geographical map using 
    COAsT and cartopy. See plot_amplitude_errors_on_map() for more info'''
    for cc in range(0,len(stats.constituent)):
        stats_tmp = stats.isel(constituent=cc)
        cmax = np.nanmean(stats_tmp.abs_error_a) + 3*np.nanstd(stats_tmp.abs_error_a)
        cmin = np.nanmean(stats_tmp.abs_error_a) - 3*np.nanstd(stats_tmp.abs_error_a)
        cmin = np.max([cmin,0])
        scatter_kwargs = {"cmap":"Reds", "marker":"o", "ec":[0,0,0], 
                          'linewidth':0.5, "vmin":cmin, "vmax":cmax}
        const_str = str(stats_tmp.constituent_name.values)
        title = 'Absolute Amplitude Error (m) | ' + run_name + ' | ' + const_str
        sca = coastpu.geo_scatter(stats.longitude, stats.latitude, 
                    c = stats_tmp.abs_error_a, s=15,
                    scatter_kwargs=scatter_kwargs, title = title)
        fn_save = run_name + "_abs_amp_error_"+const_str + '.eps'
        print("  >>>>>  Saving: "+ fn_save)
        plt.savefig(os.path.join(dn_output, fn_save))
        plt.close()
    return

def plot_proportional_amplitude_errors_on_map(stats, dn_output):
    ''' Plots proportional amplitude errors on a geographical map using 
    COAsT and cartopy. See plot_amplitude_errors_on_map() for more info'''
    for cc in range(0,len(stats.constituent)):
        stats_tmp = stats.isel(constituent=cc)
        cmax = np.nanmean(stats_tmp.prop_error_a) + 3*np.nanstd(stats_tmp.prop_error_a)
        cmin = -cmax
        cmax = np.min([cmax, 1])
        cmin = np.max([cmin,-1])
        scatter_kwargs = {"cmap":"coolwarm", "marker":"o", "ec":[0,0,0], 
                          'linewidth':0.5, "vmin":cmin, "vmax":cmax}
        const_str = str(stats_tmp.constituent_name.values)
        title = 'Proportional Amplitude Error (%) | ' + run_name + ' | ' + const_str
        sca = coastpu.geo_scatter(stats.longitude, stats.latitude, 
                    c = stats_tmp.prop_error_a, s=15,
                    scatter_kwargs=scatter_kwargs, title = title)
        fn_save = run_name + "_prop_amp_error_"+const_str + '.eps'
        print("  >>>>>  Saving: "+ fn_save)
        plt.savefig(os.path.join(dn_output, fn_save))
        plt.close()
    return
    
def write_stats_to_file(stats, dn_output):
    '''
    Writes the stats xarray dataset to a new netcdf file in the output 
    directory. stats is the dataset created by the calculate_statistics()
    routine. dn_output is the specified output directory for the script.
    '''
    fn_save = run_name + "_stats.nc"
    stats.to_netcdf(os.path.join(dn_output, fn_save))
    return

if __name__ == '__main__':
    main()