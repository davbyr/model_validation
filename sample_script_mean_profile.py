import numpy as np
import sys
import glob
from analyse_ts_monthly_en4 import analyse_ts_per_file
import os.path
import coast 

index = int(sys.argv[1])
print(index)

fn_nemo_data = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/outputs/p0/*.nc"
fn_nemo_domain = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/inputs/CO7_EXACT_CFG_FILE.nc"
fn_en4 = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/obs/en4/*.nc"
dn_out = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/analysis/tmp"
run_name = 'CO9p0 AMM15'
surface_def = 5
bottom_def = 10
dist_crit=5

fn_list = glob.glob(fn_nemo_data)
fn_out = os.path.join(dn_out, os.path.splitext(fn_list[index]) + '_out.nc')

analyse_ts_per_file(fn_list[index], fn_nemo_domain, fn_en4, fn_out, 
                    run_name, surface_def, bottom_def, dist_crit)
