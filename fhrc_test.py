#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:06:57 2023

@author: 
"""

from horz_data_divn import horz_data_divn
from frhc import frhc

df_list = horz_data_divn('diabetes', 5)

'''
    Test driver code for FHRC section of FSHFL
'''

for df in df_list:
    clusters, sc = frhc(df.iloc[:, :-1])
