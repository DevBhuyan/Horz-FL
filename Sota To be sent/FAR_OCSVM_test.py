#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:02:13 2023

@author: 
"""

from horz_data_divn import horz_data_divn
from FAR_OCSVM_data_cleaning import FAR_OCSVM_data_cleaning

df_list = horz_data_divn('diabetes', 5)

cleaned_data = FAR_OCSVM_data_cleaning(df_list)
