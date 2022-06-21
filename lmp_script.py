# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:24:06 2022

@author: hssemba
"""

import numpy as np
import pandas as pd
import os
from shutil import copy
from pathlib import Path
import math
#%matplotlib inline
import matplotlib.pyplot as plt

NODE_NUMBER = [50,75,100,125,150,175,200,225,250,275,300]
#NODE_NUMBER = [100]

UC_TREATMENTS = ['_simple']

trans_p = [25,50,75,100]
#trans_p = [100]
lmp_prices=[]
nodes=[]
for NN in NODE_NUMBER:
    
    for UC in UC_TREATMENTS:
        
        for T_p in trans_p:
            
            FN= "Exp"+ str(NN) + str(UC)+ "_" + str(T_p)
            nodes.append(FN)
            dl = FN + "/duals.csv"
            df_duals=pd.read_csv(dl)
            nl = FN + "/nodal_load.csv"
            nodal_load=pd.read_csv(nl)
            
            #rename value as duals price
            df_duals=df_duals.rename(columns={"Value": "duals_price"})
            #set the bus as an index
            df_duals = df_duals.set_index('Bus')
            
            #weight_frac reads the first row of duals to get weights based on the load
            df_duals["weight_frac"]=nodal_load.multiply(1./nodal_load.sum(axis=1), axis=0).iloc[0]
            
            #weighted dual is the weight_frac x Duals_price
            df_duals['weighted_dual']=df_duals['weight_frac']*df_duals['duals_price']
            
            #groupby the time to obtain the time series
            df_duals2=df_duals.groupby(['Time'], sort=True).agg({'duals_price':'mean','weighted_dual':'sum'}).reset_index()
            
            lmp_prices.append(df_duals2["weighted_dual"])
            #read.append(FN)
            #print(nodal_load)
            #with open(os.path.join(os.getcwd(), FN), 'r') as f:
                
df_lmp_prices=pd.DataFrame(lmp_prices)
df_lmp_prices=df_lmp_prices.transpose()
df_lmp_prices.columns= nodes            
            
                

