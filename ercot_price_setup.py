# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:15:01 2021

@author: jkern
"""

import pandas as pd

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

for m in months:
    
    df = pd.read_excel('2019_ercot_historical_LMPs.xlsx',sheet_name = m)
    sample = df.loc[df['Settlement Point']=='HB_HUBAVG','Settlement Point Price']
    
    if months.index(m) < 1:
        combined = sample
    else:
        combined = pd.concat([combined,sample])

combined = combined.reset_index(drop=True)
df_combined = pd.DataFrame(combined)
df_combined.to_csv('2019_ERCOT_hist_lmps.csv')