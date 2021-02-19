"""
@author: Dave (dbyrne@noc.ac.uk)
"""

fn_nemo_data = ""
fn_nemo_domain    = ""
fn_gesla = ""

dn_output = ""

##############################################################################

import coast
import numpy as np
import utide
import matplotlib.pyplot as plt

def main():
    print('Main')
    
    mod = read_model_ssh(fn_nemo_data, fn_nemo_domain)
    obs = read_observed_ssh(fn_gesla)


def read_model_ssh(fn_nemo_data, fn_nemo_domain ):
    nemo = coast.NEMO(fn_nemo_data, fn_nemo_domain)
    return nemo.ssh

def read_observed_ssh(fn_gesla):
    tg_list = coast.TIDEGAUGE.create_multiple_tidegauge(fn_gesla)
    return tg_list

if __name__ == '__main__':
    main()