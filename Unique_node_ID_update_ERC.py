# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:24:40 2022

@author: kakdemi
"""

import pandas as pd
import numpy as np

#Defining the interconnection number (1 for WEST, 2 for EAST, 3 for ERCOT)
Interconnection_ID = 3

#Defining number of nodes
NODE_NUMBER = [50,75,100,125,150,175,200,225,250,275,300]
NODE_NUMBER_2 = [50,75,100,125,150,175,200,225,250,275,300]

#Changing node IDs in selected, excluded nodes as well as outputs in ASU network reduction algorithm
for NN in NODE_NUMBER:
    
    #Changing selected and excluded node IDs
    selected = pd.read_csv('selected_nodes_{}.csv'.format(NN))
    excluded = pd.read_csv('excluded_nodes_{}.csv'.format(NN))
    
    selected_new = selected.copy()
    excluded_new = excluded.copy()
    
    for i in range(len(selected)):
        selected_new.loc[i,'SelectedNodes'] = int(str(selected_new.loc[i,'SelectedNodes']) + str(Interconnection_ID))
        
    for i in range(len(excluded)):
        excluded_new.loc[i,'ExcludedNodes'] = int(str(excluded_new.loc[i,'ExcludedNodes']) + str(Interconnection_ID))
        
    selected_new.to_csv('New_files_new_IDs/selected_nodes_{}.csv'.format(NN),index=False)
    excluded_new.to_csv('New_files_new_IDs/excluded_nodes_{}.csv'.format(NN),index=False)

#Since there are no reduction files for 50 nodes, I started another for loop    
for NN in NODE_NUMBER_2:
    
    #Changing node IDs in outputs of ASU network reduction algorithm
    Summary_df = pd.read_excel('Results_Excluded_Nodes_{}.xlsx'.format(NN),header=0,sheet_name='Summary')
    Bus_df = pd.read_excel('Results_Excluded_Nodes_{}.xlsx'.format(NN),header=0,sheet_name='Bus')
    Gen_df = pd.read_excel('Results_Excluded_Nodes_{}.xlsx'.format(NN),header=0,sheet_name='Gen')
    Branch_df = pd.read_excel('Results_Excluded_Nodes_{}.xlsx'.format(NN),header=0,sheet_name='Branch')
    
    Summary_df_new = Summary_df.copy()
    Bus_df_new = Bus_df.copy()
    Gen_df_new = Gen_df.copy()
    Branch_df_new = Branch_df.copy()
    
    for i in range(len(Branch_df)):
        
        Branch_df_new.loc[i,'fbus'] = int(str(Branch_df_new.loc[i,'fbus']) + str(Interconnection_ID))
        Branch_df_new.loc[i,'tbus'] = int(str(Branch_df_new.loc[i,'tbus']) + str(Interconnection_ID))
        
    for i in range(len(Gen_df)):
        
        Gen_df_new.loc[i,'Bus'] = int(str(Gen_df_new.loc[i,'Bus']) + str(Interconnection_ID))
        
    for i in range(len(Bus_df)):
        
        Bus_df_new.loc[i,'bus_i'] = int(str(Bus_df_new.loc[i,'bus_i']) + str(Interconnection_ID))
        
    for i in range(5,len(Summary_df)):
        
        selected_sentence = Summary_df_new.loc[i,'**********Reduction Summary****************']
        splitted_sentence = selected_sentence.split()
        
        splitted_sentence[4] = splitted_sentence[4] + str(Interconnection_ID)
        splitted_sentence[8] = splitted_sentence[8] + str(Interconnection_ID)
        
        joined_sentence = ' '.join(splitted_sentence)
        
        Summary_df_new.loc[i,'**********Reduction Summary****************'] = joined_sentence
        
    with pd.ExcelWriter('New_files_new_IDs/Results_Excluded_Nodes_{}.xlsx'.format(NN), engine='openpyxl') as writer:  
        Summary_df_new.to_excel(writer, sheet_name='Summary',index=False)
        Bus_df_new.to_excel(writer, sheet_name='Bus',index=False)
        Gen_df_new.to_excel(writer, sheet_name='Gen',index=False)
        Branch_df_new.to_excel(writer, sheet_name='Branch',index=False)
        
#Changing nodes_to_BA_state
nodes_to_BA = pd.read_csv('nodes_to_BA_state.csv',header=0)
del nodes_to_BA['Unnamed: 0']
nodes_to_BA_new = nodes_to_BA.copy()

for i in range(len(nodes_to_BA)):
    nodes_to_BA_new.loc[i,'Number'] = int(str(nodes_to_BA_new.loc[i,'Number']) + str(Interconnection_ID))
    
nodes_to_BA_new.to_csv('New_files_new_IDs/nodes_to_BA_state.csv')
    
    
#Changing NG_coal_heat rates
NG_coal_heat = pd.read_excel('NG_Coal_heat_rates.xlsx',header=0)
NG_coal_heat_new = NG_coal_heat.copy()

for i in range(len(NG_coal_heat)):
    NG_coal_heat_new.loc[i,'BusNum'] = int(str(NG_coal_heat_new.loc[i,'BusNum']) + str(Interconnection_ID))
    
NG_coal_heat_new.to_excel('New_files_new_IDs/NG_Coal_heat_rates.xlsx',index=False)   
    
    
#Changing EIA hydro
EIA_hydro_plants = pd.read_csv('EIA_302_WECC_hydro_plants.csv',header=0)
EIA_hydro_plants_new = EIA_hydro_plants.copy()

for i in range(len(EIA_hydro_plants)):
    EIA_hydro_plants_new.loc[i,'bus'] = int(str(EIA_hydro_plants_new.loc[i,'bus']) + str(Interconnection_ID))
    
EIA_hydro_plants_new.to_csv('New_files_new_IDs/EIA_302_WECC_hydro_plants.csv',index=False)    

             
#Changing 10k_load
Load_TAMU = pd.read_csv('10k_Load.csv',header=0)
Load_TAMU_new = Load_TAMU.copy()

for i in range(len(Load_TAMU)):
    Load_TAMU_new.loc[i,'Number'] = int(str(Load_TAMU_new.loc[i,'Number']) + str(Interconnection_ID))
    
Load_TAMU_new.to_csv('New_files_new_IDs/10k_Load.csv',index=False)   
        
          
#Changing 10k_heatrate
Heatrates_TAMU = pd.read_csv('10k_Heat rates.csv',header=1)
Heatrates_TAMU_new = Heatrates_TAMU.copy()

for i in range(len(Heatrates_TAMU)):
    Heatrates_TAMU_new.loc[i,'BusNum'] = int(str(Heatrates_TAMU_new.loc[i,'BusNum']) + str(Interconnection_ID))
    
Heatrates_TAMU_new.to_csv('New_files_new_IDs/10k_Heat rates.csv',index=False)           
        

#Changing 10k_gen
Gen_TAMU = pd.read_csv('10k_Gen.csv',header=0)
Gen_TAMU_new = Gen_TAMU.copy()

for i in range(len(Gen_TAMU)):
    Gen_TAMU_new.loc[i,'BusNum'] = int(str(Gen_TAMU_new.loc[i,'BusNum']) + str(Interconnection_ID))
    
Gen_TAMU_new.to_csv('New_files_new_IDs/10k_Gen.csv',index=False)          
        
    
    
    
    
        
        
    







