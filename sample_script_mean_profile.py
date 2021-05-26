import numpy as np
import sys
import os

# CHANGE: SET THIS TO BE THE MODEL VALIDATION CODE DIRECTORY
os.chdir('/home/users/dbyrne/code/mv2/')

# CHANGE: IF USING A DEVELOPMENT VERSION OF COAST UNCOMMENT THIS LINE
sys.path.append('/home/users/dbyrne/code/COAsT')

import glob
from analyse_ts_monthly_en4 import analyse_ts_per_file
import os.path
import coast 
from dateutil.relativedelta import *
from datetime import datetime

# ROUTINE FOR MAKING NEMO FILENAME FROM DATETIME OBJECT.
# CHANGE: MODIFY IF THIS DOESNT MATCH YOUR NAMING CONVENTION
def make_nemo_filename(dn, dt):
    suffix = '_daily_grid_T'
    day = str(dt.day).zfill(2) 
    month = str(dt.month).zfill(2)
    year = dt.year
    yearmonth = str(year) + str(month) + day
    return os.path.join(dn, yearmonth + suffix + '.nc')

# ROUTINE FOR MAKING EN4 FILENAME FROM DATETIME OBJECT.
# CHANGE: MODIFY IF THIS DOESNT MATCH YOUR NAMING CONVENTION
def make_en4_filename(dn, dt):
    prefix = 'EN.4.2.1.f.profiles.g10.'
    month = str(dt.month).zfill(2)
    year = dt.year
    yearmonth = str(year) + str(month)
    return os.path.join(dn, prefix + yearmonth + '.nc')

# CHANGE: START AND END DATES FOR ANALYSIS
start_date = datetime(2004,1,1) 
end_date = datetime(2014,12,1)

# CHANGE: DIRECTORIES AND FILE PATHS FOR ANALYSIS.
# NEMO data directory and domain file
dn_nemo_data = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/outputs/daily/p0/"
fn_nemo_domain = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/inputs/CO7_EXACT_CFG_FILE.nc"
# EN4 directory
dn_en4 = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/obs/en4/"
# Output directory
dn_out = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/analysis/tmp"
# Name of this run
run_name = 'CO9p0 AMM15'
# Surface and bottom definitions
surface_def = 5
bottom_def = 10
# Interpolation distance at which to omit data
dist_crit=5

##################################################################

# Start of MAIN script

##################################################################

# Get input from command line
print(str(sys.argv[1]), flush=True)
index = int(sys.argv[1])
print(index)

n_months = (end_date.year - start_date.year)*12 + \
           (end_date.month - start_date.month) + 1
month_list = [start_date + relativedelta(months=+mm) for mm in range(0,n_months)]

nemo_filename = make_nemo_filename(dn_nemo_data, month_list[index])
en4_filename = make_en4_filename(dn_en4, month_list[index])

print(nemo_filename)
print(en4_filename)

fn_out = os.path.join(dn_out, os.path.splitext(os.path.basename(nemo_filename))[0] + '_out.nc')

analyse_ts_per_file(nemo_filename, fn_nemo_domain, en4_filename, fn_out, 
                    run_name, surface_def, bottom_def, dist_crit)
