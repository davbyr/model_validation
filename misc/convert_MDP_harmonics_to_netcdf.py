#Script for reading in the MDP global harmonic dataset txtfile

import numpy as np
import xarray as xr

####################
####################

fn_in = '/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/MDP_harmonics.txt' # Input file location
fn_out = '/Users/Dave/Documents/Projects/CO9_AMM15/validation/data/MDP_harmonics.nc'


####################
####################

n_const = 115

#Open and read all of file.
f_id = open(fn_in)
all_lines = f_id.readlines()

#Determine number of lines/locations
n_ports = len(all_lines)

#Initialise datasets.
data_type        = np.zeros(n_ports) * np.nan
latitude         = np.zeros(n_ports) * np.nan
longitude        = np.zeros(n_ports) * np.nan
z0               = np.zeros(n_ports) * np.nan
port_name        = np.zeros(n_ports, dtype=str)

amplitude = np.zeros((n_const, n_ports)) * np.nan
phase     = np.zeros((n_const, n_ports)) * np.nan

#Read  through each line, putting data into the expected arrays.
line_ii = -1
for line in all_lines:
    
    line_ii = line_ii + 1   
    line0                  = line[0:30].split()

    data_type[line_ii]     = float(line0[0]) 
    latitude[line_ii]      = float(line0[1]) 
    longitude[line_ii]     = float(line0[2]) 
    z0[line_ii]            = float(line0[3]) 
    port_name[line_ii]     = str(line[30:62]).strip()
    
    line0 = line[63:].split()
    for const_ii in range(0, len(line0), 3):
        doodson_index = int(line0[const_ii + 2])
        amplitude[doodson_index-1, line_ii] = float(line0[const_ii])
        phase[doodson_index-1, line_ii]     = float(line0[const_ii + 1])

# Close open file
f_id.close()

#Convert all to numpy arrays
data_type  = np.array(data_type)
port_name  = np.array(port_name)
latitude   = np.array(latitude) 
longitude  = np.array(longitude)
z0         = np.array(z0)
doodson_index = np.arange(1,n_const+1)
n_ports = len(latitude)

# Constituent name list -- aligns with doodson indices
constituent_names = ['SA','SSA','MM','MSF','MF','2Q1','SIGMA1','Q1','RHO1','O1',
                     'MP1','M1','CHI1','PI1','P1','S1','K1','PSI1','PHI1',
                     'THETA1','J1','SO1','OO1','OQ2','MNS2','2N2','MU2','N2',
                     'NU2','OP2','M2','MKS2','LAMBDA2','L2','T2','S2','R2','K2',
                     'MSN2','KJ2','2SM2','MO3','M3','SO2','MK3','SK3','MN4',
                     'M4','SN4','MS4','MK4','S4','SK4','2MN6','M6','MSN6',
                     '2MS6','2MK6','2SM6','MSK6','2MN2S2','3MSK2','3M2S2',
                     'MNK2S2','SNK2','2SK2','2MS2N2','MQ3','2MP3','2MQ3',
                     '3MK4','3MS4','2MSK4','3MK5','M5','3MO5','2MNS6','3MNS6',
                     '4MK6','4MS6','2MSNK6','2MV6','3MSK6','4MN6','3MSN6',
                     'MKL6','2MN8','3MN8','M8','2MSN8','3MS8','3MK8','MSNK8',
                     '2MS8','2MSK8','4MS10','3M2S10','4MSN12','5MS12','4M2S12',
                     'MVS2','2MK2','MA2','MB2','MSV2','SKM2','2MNS4','MV4',
                     '3MN4','2MSN4','NA2','NB2','MSO5','MSK5','2MN2']
constituent_names = np.array(constituent_names, dtype='str')

#Change interval phase lies in
phase = phase - 360*(phase>180)    

# Remove any constituents which are nan at all ports
nan_indices = ~np.isnan(amplitude).all(axis=1)
doodson_index = doodson_index[nan_indices]
constituent_names = constituent_names[nan_indices]
amplitude     = amplitude[nan_indices,:]
phase         = phase[nan_indices,:]

# Create xarray Dataset
description = "RESTRICTED DATA. Make sure you have permission to use and (especially) share. + "\
    "Tidal Harmonics from NOC MDP, converted from ASCII file. "+\
        "Converted by David Byrne. "
dataset = xr.Dataset(data_vars = dict(
                         amplitude=(["constituent","port"], amplitude),
                         phase = (["constituent","port"], phase),
                         z0 = (["port"], z0),
                         data_type_flag = (["port"], data_type),
                     ),
                     coords = dict(
                         longitude=(["port"], longitude),
                         latitude=(["port"], latitude),
                         constituent_name=(["constituent"], constituent_names),
                         doodson_index=(["constituent"],doodson_index)
                     ),
                     attrs = dict(description=description))
desc_data_type = " Describes the data source. 1 is the quality controlled MDP " +\
                 " data."
dataset.data_type_flag.attrs = dict(description=desc_data_type)

# Write dataset to file
dataset.to_netcdf(fn_out)