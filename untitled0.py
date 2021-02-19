#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:56:50 2021

@author: dbyrne
"""

import coast
import coast.general_utils as general_utils
import numpy as np
import xarray as xr

fn_nemo_dat = ""
fn_nemo_domain = ""
fn_out = ""

extract_latitude, extract_longitude = locations_to_extract()

model = read_model_nem(fn_nemo_dat, fn_nemo_domain, grid_ref='t-grid')

ind2D = general_utils.nearest_indices_2D()



def locations_to_extract():
    return

def read_model_nemo(fn_nemo_dat, fn_nemo_domain, grid_ref):
    model = coast.NEMO(fn_nemo_dat, fn_nemo_domain, grid_ref = grid_ref)
    return