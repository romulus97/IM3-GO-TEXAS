# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:15:02 2021

@author: kakdemi
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

#reading BA names and creating a date range for 2019
BA_name_data = pd.read_csv('BAs.csv',header=0)
BA_abb = BA_name_data['Abbreviation']
BA_names = BA_name_data['Name']
days_2019 = pd.date_range(start='2019-01-01',end='2019-12-31',freq='D')
hours_2019 = pd.date_range(start='2019-01-01 00:00:00',end='2019-12-31 23:00:00',freq='H')

#calculating historical % contribution of each generation type annually, and saving daily generation from each generation type
for BA in BA_abb:
    #reading historical generation data and filling empyt values with 0
    globals()[BA] = pd.read_excel('../Raw_Data/{}.xlsx'.format(BA), sheet_name='Published Daily Data',header=0,parse_dates=True)
    globals()[BA].fillna(0,inplace=True)
    
    #filtering data and getting only 2019 values
    globals()[BA].set_index('Local date',drop=True,inplace=True)
    globals()[BA] = globals()[BA].loc[days_2019,:]
   
    #getting daily generation from each type and calculating annual % contribution
    globals()[BA+'_hist_daily'] = globals()[BA].loc[:,['NG','NG: COL','NG: NG','NG: NUC','NG: OIL','NG: WAT','NG: SUN','NG: WND' ]]
    globals()[BA+'_hist_daily'].columns = ['Netgen','Coal','Gas','Nuclear','Oil','Hydro','Solar','Wind']
    globals()[BA+'_hist_yearly'] = globals()[BA+'_hist_daily'].sum(axis=0).copy()
    globals()[BA+'_hist_yearly'] = globals()[BA+'_hist_yearly']/globals()[BA+'_hist_yearly']['Netgen']*100
    
    #checking if there is any missing data
    if globals()[BA+'_hist_daily'].isna().sum().sum() > 0:
        print('{} has some missing data.'.format(BA))
    else:
        pass
    

#reading bus to BA data
bus_to_BA = pd.read_csv('nodes_to_BA_state.csv', header=0, usecols = ['Number','NAME'])
    
#defining generation mix validation function
def GenMix_validation(sim_details):
    
    #reading necessary data
    mwh = pd.read_csv('Model_results/mwh{}.csv'.format(sim_details), header=0, usecols=['Generator','Type','Time','Value'])
    genparams = pd.read_csv('Model_results/data_genparams{}.csv'.format(sim_details), header=0, usecols=['name','typ','node'])
    must_run = pd.read_csv('Model_results/must_run{}.csv'.format(sim_details), header=0)
    
    #creating daily arrays to store BA sim generation mix data
    for BA in BA_abb:
        globals()['{}_sim_daily'.format(BA)] = np.zeros((365,6))
    
    for gen in mwh['Generator'].unique():
        
        #getting daily generation for each generator and finding node number
        mwh_gen = mwh.loc[mwh['Generator']==gen].copy()
        gen_type = mwh_gen['Type'].values[0]
        mwh_gen.index = hours_2019
        mwh_gen = mwh_gen.resample('D').sum()
        mwh_gen = np.array(mwh_gen['Value'])
        
        if gen_type == 'Coal' or gen_type == 'Gas':   
            node_num = int(genparams.loc[genparams['name']==gen]['node'].values[0][4:])
                
        elif gen_type == 'Wind' :
            node_num = int(gen[4:-5])
               
        elif gen_type == 'Solar' or gen_type == 'Hydro':
            node_num = int(gen[4:-6])
                
        else:
            pass
        
        #finding BA of each generator
        node_BA = bus_to_BA[bus_to_BA['Number']==node_num]['NAME'].values[0]
        node_BA = BA_name_data.loc[BA_name_data['Name']==node_BA]['Abbreviation'].values[0]
    
        #appending daily generation with respect to generator type
        if gen_type == 'Coal':
            
            if np.sum(globals()['{}_sim_daily'.format(node_BA)][:,0]) == 0:
                globals()['{}_sim_daily'.format(node_BA)][:,0] = mwh_gen
            else:
                globals()['{}_sim_daily'.format(node_BA)][:,0] = globals()['{}_sim_daily'.format(node_BA)][:,0] + mwh_gen
                
        elif gen_type == 'Gas':
            
            if np.sum(globals()['{}_sim_daily'.format(node_BA)][:,1]) == 0:
                globals()['{}_sim_daily'.format(node_BA)][:,1] = mwh_gen
            else:
                globals()['{}_sim_daily'.format(node_BA)][:,1] = globals()['{}_sim_daily'.format(node_BA)][:,1] + mwh_gen
               
        elif gen_type == 'Solar':
            
            if np.sum(globals()['{}_sim_daily'.format(node_BA)][:,2]) == 0:
                globals()['{}_sim_daily'.format(node_BA)][:,2] = mwh_gen
            else:
                globals()['{}_sim_daily'.format(node_BA)][:,2] = globals()['{}_sim_daily'.format(node_BA)][:,2] + mwh_gen
            
        elif gen_type == 'Wind':
            
            if np.sum(globals()['{}_sim_daily'.format(node_BA)][:,3]) == 0:
                globals()['{}_sim_daily'.format(node_BA)][:,3] = mwh_gen
            else:
                globals()['{}_sim_daily'.format(node_BA)][:,3] = globals()['{}_sim_daily'.format(node_BA)][:,3] + mwh_gen
                
        elif gen_type == 'Hydro':
            
            if np.sum(globals()['{}_sim_daily'.format(node_BA)][:,4]) == 0:
                globals()['{}_sim_daily'.format(node_BA)][:,4] = mwh_gen
            else:
                globals()['{}_sim_daily'.format(node_BA)][:,4] = globals()['{}_sim_daily'.format(node_BA)][:,4] + mwh_gen  
        
        else:
            pass
                
    #finding nuclear generation from mustrun files and appending those
    for node in must_run.columns:
        
        node_num = int(node[4:])
        node_nuclear = must_run[node][0]
        
        node_BA = bus_to_BA[bus_to_BA['Number']==node_num]['NAME'].values[0]
        node_BA = BA_name_data.loc[BA_name_data['Name']==node_BA]['Abbreviation'].values[0]
        
        node_nuclear = np.repeat(node_nuclear*24, 365)
        
        if np.sum(globals()['{}_sim_daily'.format(node_BA)][:,5]) == 0:
            globals()['{}_sim_daily'.format(node_BA)][:,5] = node_nuclear
        else:
            globals()['{}_sim_daily'.format(node_BA)][:,5] = globals()['{}_sim_daily'.format(node_BA)][:,5] + node_nuclear   
        
    #finding simulated % contribution of each generation type annually
    for BA in BA_abb:
        
        globals()['{}_sim_daily'.format(BA)] = pd.DataFrame(globals()['{}_sim_daily'.format(BA)],\
                                                                 columns = ['Coal','Gas','Solar','Wind','Hydro','Nuclear'])
        
        globals()[BA+'_sim_yearly'] = globals()[BA+'_sim_daily'].sum(axis=0).copy()
        globals()[BA+'_sim_yearly'] = globals()[BA+'_sim_yearly']/globals()[BA+'_sim_yearly'].sum()*100
        
    #organizing yearly generation mix percentages and RMSE values for each generation type
    yearly_percentages = []
    all_RMSE_vals = []
        
    for source in ['Coal','Gas','Solar','Wind','Hydro','Nuclear']:
        
        hist_yearly_data = []
        sim_yearly_data = []
        RMSE_vals = []
        
        for BA in BA_abb:
            
            RMSE = mean_squared_error(globals()['{}_hist_daily'.format(BA)][source], globals()['{}_sim_daily'.format(BA)][source], squared=False)
            RMSE_vals.append(RMSE)
            hist_yearly_data.append(globals()[BA+'_hist_yearly'][source])
            sim_yearly_data.append(globals()[BA+'_sim_yearly'][source])
        
        all_RMSE_vals.append(RMSE_vals)    
        yearly_percentages.append(hist_yearly_data)
        yearly_percentages.append(sim_yearly_data)
                 
    return yearly_percentages, all_RMSE_vals
            
           
#defining different model types  and a empty list to store data
# node_numbers = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
# UC_treatment = ['_simple_','_coal_']
# transmission_multiplier = [25, 50, 75, 100]

node_numbers = [100]
UC_treatment = ['simple','coal']
transmission_multiplier = [25]

#running the generation mix validation for each model type and saving results
i=0
for NN in node_numbers:
    i+=1
    
    for UC in UC_treatment:
        i+=1
        
        for TP in transmission_multiplier:
            i+=1
            
            sim_details = '_{}_{}_{}'.format(NN,UC,TP)
            yearly_percent, RMSE_values = GenMix_validation(sim_details) 
            
            Daily_RMSE = pd.DataFrame(RMSE_values, columns=BA_abb)
            Daily_RMSE.index = ['Coal','Gas','Solar','Wind','Hydro','Nuclear']
            
            Yearly_gen_mix = pd.DataFrame(yearly_percent, columns=BA_abb)
            Yearly_gen_mix.index = ['Coal_Hist','Coal_Sim','Gas_Hist','Gas_Sim','Solar_Hist','Solar_Sim','Wind_Hist','Wind_Sim','Hydro_Hist','Hydro_Sim','Nuclear_Hist','Nuclear_Sim']
            
            if i==3:

                Yearly_gen_mix.to_excel('Yearly_gen_mix_comparison.xlsx', sheet_name=sim_details[1:])
                Daily_RMSE.to_excel('Daily_genmix_RMSE.xlsx', sheet_name=sim_details[1:])
                
            else:

                with pd.ExcelWriter('Yearly_gen_mix_comparison.xlsx',mode='a') as writer:  
                    Yearly_gen_mix.to_excel(writer, sheet_name=sim_details[1:])
                    
                with pd.ExcelWriter('Daily_genmix_RMSE.xlsx',mode='a') as writer:  
                    Daily_RMSE.to_excel(writer, sheet_name=sim_details[1:])
                    
                



