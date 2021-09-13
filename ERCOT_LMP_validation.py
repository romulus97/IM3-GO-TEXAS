# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 20:32:00 2021

@author: kakdemi
"""

import pandas as pd
import numpy as np
import datetime
# import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from matplotlib.lines import Line2D

#preparing 2019 LMP data for MidC and Palo Verde (reading the data and interpolating the missing LMPs)
historical_LMP_2019 = pd.read_csv('2019_ERCOT_hist_lmps.csv', header=0,index_col=0)
hours_2019 = pd.date_range(start='1-1-2019 00:00:00',end='12-31-2019 23:00:00', freq='H')
historical_LMP_2019.index = hours_2019
daily_historical_LMP = historical_LMP_2019.resample('D').mean()
daily_historical_LMP.columns = ['Price']

  
######### 1st option: mean of all nodes within a zone #########

#empty list to store R2s
R2_mean_method = []

# #initilizing a figure
# plt.rcParams.update({'font.size': 30})
# plt.rcParams['font.sans-serif'] = "Arial"
# plt.style.use('seaborn-whitegrid')

# axis_fontsize=33
# axis_label_pad = 10
# tick_pad = 10

# fig, ax = plt.subplots(2,5 , figsize=(60,20))

#calculating R2 for every zone and plotting line and scatter plots to compare

#defining different model types  and a empty list to store data
node_numbers = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
UC_treatment = ['simple','coal']
transmission_multiplier = [25, 50, 75, 100]

R2_data = []
count = 0

plt.figure()

#running the LMP validation for each model type and saving R2 values
for NN in node_numbers:
    
    for UC in UC_treatment:
        
        for TP in transmission_multiplier:
            
            sim_LMP_name = 'results/duals_Exp{}_{}_{}.csv'.format(NN,UC,TP)
            sim_LMP = pd.read_csv(sim_LMP_name, header=0)
            hourly_simulated_LMP = sim_LMP.groupby('Time').mean()
            
            #temp fix
            end_prices = hourly_simulated_LMP[8712:].copy()
            end_prices.index = range(8737,8761)
            
            hourly_simulated_LMP = hourly_simulated_LMP.append(end_prices)
            
            hours_2019 = pd.date_range(start='1-1-2019 00:00:00',end='12-31-2019 23:00:00', freq='H')
            hourly_simulated_LMP.index = hours_2019
            daily_simulated_LMP = hourly_simulated_LMP.resample('D').mean()
            daily_simulated_LMP.columns = ['Price']
  
            daily_reg = linear_model.LinearRegression()
            daily_reg.fit(daily_historical_LMP['Price'].values.reshape(-1, 1), daily_simulated_LMP['Price'].values.reshape(-1, 1))
            R2 = daily_reg.score(daily_historical_LMP['Price'].values.reshape(-1, 1), daily_simulated_LMP['Price'].values.reshape(-1, 1))
            R2_data.append(R2)
            
            plt.plot(daily_simulated_LMP['Price'])


plt.plot(daily_historical_LMP['Price'])

idx = np.where(R2_data==max(R2_data))
            
plt.savefig('ercot.png',dpi=300)


# sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=daily_average_LMP['{}_LMP'.format(zone)], ax=ax[0,idx], color='black')
# sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[0,idx], color='crimson')
# ax[0,idx].set_xlabel("Date", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
# if idx == 0:
#     ax[0,idx].set_ylabel("LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
# else:
#     ax[0,idx].set_ylabel('')
# ax[0,idx].set_xticks(['2019-01-01', '2019-05-01', '2019-09-01', '2019-12-15']) 
# ax[0,idx].set_xticklabels(['Jan 2019', 'May 2019', 'Sep 2019', 'Dec 2019']) 
# # ax[0,idx].set_yticks([0,20,40,60,80,100,120,140,160,180])
# ax[0,idx].tick_params(axis='both', which='both', pad=tick_pad)
# ax[0,idx].set_title(zone)
# ax[0,idx].annotate('$R^2$ = {}'.format(round(R2,2)), xy=(0.7, 0.9), xycoords='axes fraction', fontsize=40)

# sns.regplot(x=daily_average_LMP['{}_LMP'.format(zone)], y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[1,idx], color='royalblue', ci=None)
# ax[1,idx].set_xlabel("Simulated LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
# if idx == 0:
#     ax[1,idx].set_ylabel("Historical LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
# else:
#     ax[1,idx].set_ylabel('')
# ax[1,idx].tick_params(axis='both', which='both', pad=tick_pad)
    
# #creating a custom legend for the figure
# labels = []
# handles = []

# legend1 = Line2D([], [], color='black', linewidth=3)
# handles.insert(0,legend1)
# labels.insert(0,'Simulated')

# legend2 = Line2D([], [], color='crimson', linewidth=3)
# handles.insert(1,legend2)
# labels.insert(1,'Historical')

# plt.suptitle(sim_LMP_name[20:-4]+'_simple_mean', fontsize=45, ha='center', weight = 'bold')
# fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.16, 0.75), frameon=True,fontsize=35)
# plt.tight_layout()
# plt.savefig('LMP_Plots/LMP_{}_basic_mean.png'.format(sim_LMP_name[20:-4]), dpi=100, bbox_inches='tight')
# plt.show()
# plt.clf()
    

# ######### 2nd option: weighted average with respect to demand at nodes #########
      
# #empty list to store R2s
# R2_weighted_average_demand = []

# #initilizing a figure
# plt.rcParams.update({'font.size': 30})
# plt.rcParams['font.sans-serif'] = "Arial"
# plt.style.use('seaborn-whitegrid')

# axis_fontsize=33
# axis_label_pad = 10
# tick_pad = 10

# fig, ax = plt.subplots(2,5 , figsize=(60,20))

# #calculating R2 for every zone and plotting line and scatter plots to compare
# for zone in different_zones:
    
#     wght_avg_demand_LMP_total = np.zeros((len(hours_2019)))
#     buses_demand = []
    
#     for bus in locals()['{}_buses'.format(zone)]:
        
#         bus_number = int(bus[4:])
#         zonal_prices = sim_LMP.loc[sim_LMP['Bus']==bus]['Value'].values
#         df_load_10k.fillna(0,inplace=True)
#         bus_demand = df_load_10k.loc[df_load_10k['Number']==bus_number]['Load MW'].values[0]
#         if bus_demand == np.nan or bus_demand == 'nan' or bus_demand == '':
#             bus_demand = 0
#         buses_demand.append(bus_demand)
#         wght_avg_demand_LMP_total = wght_avg_demand_LMP_total + zonal_prices*bus_demand
        
#     wght_avg_demand_LMP_final = wght_avg_demand_LMP_total/sum(buses_demand)
    
    
#     hourly_average_LMP = pd.DataFrame(wght_avg_demand_LMP_final)
#     hourly_average_LMP.index = hours_2019
#     daily_average_LMP = hourly_average_LMP.resample('D').mean()
#     daily_average_LMP.columns = ['{}_LMP'.format(zone)]
    
#     daily_reg = linear_model.LinearRegression()
#     daily_reg.fit(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
#     R2 = daily_reg.score(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
#     R2_weighted_average_demand.append(R2)
#     idx = different_zones.index(zone)
    
#     sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=daily_average_LMP['{}_LMP'.format(zone)], ax=ax[0,idx], color='black')
#     sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[0,idx], color='crimson')
#     ax[0,idx].set_xlabel("Date", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     if idx == 0:
#         ax[0,idx].set_ylabel("LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     else:
#         ax[0,idx].set_ylabel('')
#     ax[0,idx].set_xticks(['2019-01-01', '2019-05-01', '2019-09-01', '2019-12-15']) 
#     ax[0,idx].set_xticklabels(['Jan 2019', 'May 2019', 'Sep 2019', 'Dec 2019']) 
#     # ax[0,idx].set_yticks([0,20,40,60,80,100,120,140,160,180])
#     ax[0,idx].tick_params(axis='both', which='both', pad=tick_pad)
#     ax[0,idx].set_title(zone)
#     ax[0,idx].annotate('$R^2$ = {}'.format(round(R2,2)), xy=(0.7, 0.9), xycoords='axes fraction', fontsize=40)

#     sns.regplot(x=daily_average_LMP['{}_LMP'.format(zone)], y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[1,idx], color='royalblue', ci=None)
#     ax[1,idx].set_xlabel("Simulated LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     if idx == 0:
#         ax[1,idx].set_ylabel("Historical LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     else:
#         ax[1,idx].set_ylabel('')
#     ax[1,idx].tick_params(axis='both', which='both', pad=tick_pad)
    
# #creating a custom legend for the figure
# labels = []
# handles = []

# legend1 = Line2D([], [], color='black', linewidth=3)
# handles.insert(0,legend1)
# labels.insert(0,'Simulated')

# legend2 = Line2D([], [], color='crimson', linewidth=3)
# handles.insert(1,legend2)
# labels.insert(1,'Historical')

# plt.suptitle(sim_LMP_name[20:-4]+'_wght_avg_demand', fontsize=45, ha='center', weight = 'bold')
# fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.16, 0.75), frameon=True,fontsize=35)
# plt.tight_layout()
# plt.savefig('LMP_Plots/LMP_{}_wght_avg_demand.png'.format(sim_LMP_name[20:-4]), dpi=100, bbox_inches='tight')
# plt.show()
# plt.clf()


# ######### 3rd option: weighted average with respect to generation capacity at nodes #########
      
# #empty list to store R2s
# R2_weighted_average_gen = []

# #initilizing a figure
# plt.rcParams.update({'font.size': 30})
# plt.rcParams['font.sans-serif'] = "Arial"
# plt.style.use('seaborn-whitegrid')

# axis_fontsize=33
# axis_label_pad = 10
# tick_pad = 10

# fig, ax = plt.subplots(2,5 , figsize=(60,20))

# #calculating R2 for every zone and plotting line and scatter plots to compare
# for zone in different_zones:
    
#     wght_avg_gen_LMP_total = np.zeros((len(hours_2019)))
#     buses_gen_cap = []
    
#     for bus in locals()['{}_buses'.format(zone)]:
        
#         bus_number = int(bus[4:])
#         zonal_prices = sim_LMP.loc[sim_LMP['Bus']==bus]['Value'].values

#         try:
#             bus_gen = gen_cap_node_all.loc[bus,'maxcap']
#         except KeyError:
#             bus_gen = 0
            
#         if bus_gen == np.nan or bus_gen == 'nan' or bus_gen == '':
#             bus_gen = 0
#         buses_gen_cap.append(bus_gen)
#         wght_avg_gen_LMP_total = wght_avg_gen_LMP_total + zonal_prices*bus_gen
        
#     wght_avg_gen_LMP_final = wght_avg_gen_LMP_total/sum(buses_gen_cap)
    
    
#     hourly_average_LMP = pd.DataFrame(wght_avg_gen_LMP_final)
#     hourly_average_LMP.index = hours_2019
#     daily_average_LMP = hourly_average_LMP.resample('D').mean()
#     daily_average_LMP.columns = ['{}_LMP'.format(zone)]
    
#     daily_reg = linear_model.LinearRegression()
#     daily_reg.fit(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
#     R2 = daily_reg.score(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
#     R2_weighted_average_gen.append(R2)
#     idx = different_zones.index(zone)
    
#     sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=daily_average_LMP['{}_LMP'.format(zone)], ax=ax[0,idx], color='black')
#     sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[0,idx], color='crimson')
#     ax[0,idx].set_xlabel("Date", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     if idx == 0:
#         ax[0,idx].set_ylabel("LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     else:
#         ax[0,idx].set_ylabel('')
#     ax[0,idx].set_xticks(['2019-01-01', '2019-05-01', '2019-09-01', '2019-12-15']) 
#     ax[0,idx].set_xticklabels(['Jan 2019', 'May 2019', 'Sep 2019', 'Dec 2019']) 
#     # ax[0,idx].set_yticks([0,20,40,60,80,100,120,140,160,180])
#     ax[0,idx].tick_params(axis='both', which='both', pad=tick_pad)
#     ax[0,idx].set_title(zone)
#     ax[0,idx].annotate('$R^2$ = {}'.format(round(R2,2)), xy=(0.7, 0.9), xycoords='axes fraction', fontsize=40)

#     sns.regplot(x=daily_average_LMP['{}_LMP'.format(zone)], y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[1,idx], color='royalblue', ci=None)
#     ax[1,idx].set_xlabel("Simulated LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     if idx == 0:
#         ax[1,idx].set_ylabel("Historical LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     else:
#         ax[1,idx].set_ylabel('')
#     ax[1,idx].tick_params(axis='both', which='both', pad=tick_pad)
    
# #creating a custom legend for the figure
# labels = []
# handles = []

# legend1 = Line2D([], [], color='black', linewidth=3)
# handles.insert(0,legend1)
# labels.insert(0,'Simulated')

# legend2 = Line2D([], [], color='crimson', linewidth=3)
# handles.insert(1,legend2)
# labels.insert(1,'Historical')

# plt.suptitle(sim_LMP_name[20:-4]+'_wght_avg_gen_cap', fontsize=45, ha='center', weight = 'bold')
# fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.16, 0.75), frameon=True,fontsize=35)
# plt.tight_layout()
# plt.savefig('LMP_Plots/LMP_{}_wght_avg_gen_cap.png'.format(sim_LMP_name[20:-4]), dpi=100, bbox_inches='tight')
# plt.show()
# plt.clf()


# ######### 4th option: weighted average with respect to generation capacity + demand at nodes #########
      
# #empty list to store R2s
# R2_weighted_avg_sum_gen_demand = []

# #initilizing a figure
# plt.rcParams.update({'font.size': 30})
# plt.rcParams['font.sans-serif'] = "Arial"
# plt.style.use('seaborn-whitegrid')

# axis_fontsize=33
# axis_label_pad = 10
# tick_pad = 10

# fig, ax = plt.subplots(2,5 , figsize=(60,20))

# #calculating R2 for every zone and plotting line and scatter plots to compare
# for zone in different_zones:
    
#     wght_avg_gen_demand_LMP_total = np.zeros((len(hours_2019)))
#     buses_gen_cap_demand = []
    
#     for bus in locals()['{}_buses'.format(zone)]:
        
#         bus_number = int(bus[4:])
#         zonal_prices = sim_LMP.loc[sim_LMP['Bus']==bus]['Value'].values

#         try:
#             bus_gen = gen_cap_node_all.loc[bus,'maxcap']
#         except KeyError:
#             bus_gen = 0
            
#         if bus_gen == np.nan or bus_gen == 'nan' or bus_gen == '':
#             bus_gen = 0
            
#         df_load_10k.fillna(0,inplace=True)
#         bus_demand = df_load_10k.loc[df_load_10k['Number']==bus_number]['Load MW'].values[0]
#         if bus_demand == np.nan or bus_demand == 'nan' or bus_demand == '':
#             bus_demand = 0
        
#         bus_info_weight = bus_demand + bus_gen
#         buses_gen_cap_demand.append(bus_info_weight)
#         wght_avg_gen_demand_LMP_total = wght_avg_gen_demand_LMP_total + zonal_prices*bus_info_weight
        
#     wght_avg_gen_demand_LMP_final = wght_avg_gen_demand_LMP_total/sum(buses_gen_cap_demand)
    
    
#     hourly_average_LMP = pd.DataFrame(wght_avg_gen_demand_LMP_final)
#     hourly_average_LMP.index = hours_2019
#     daily_average_LMP = hourly_average_LMP.resample('D').mean()
#     daily_average_LMP.columns = ['{}_LMP'.format(zone)]
    
#     daily_reg = linear_model.LinearRegression()
#     daily_reg.fit(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
#     R2 = daily_reg.score(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
#     R2_weighted_avg_sum_gen_demand.append(R2)
#     idx = different_zones.index(zone)
    
#     sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=daily_average_LMP['{}_LMP'.format(zone)], ax=ax[0,idx], color='black')
#     sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[0,idx], color='crimson')
#     ax[0,idx].set_xlabel("Date", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     if idx == 0:
#         ax[0,idx].set_ylabel("LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     else:
#         ax[0,idx].set_ylabel('')
#     ax[0,idx].set_xticks(['2019-01-01', '2019-05-01', '2019-09-01', '2019-12-15']) 
#     ax[0,idx].set_xticklabels(['Jan 2019', 'May 2019', 'Sep 2019', 'Dec 2019']) 
#     # ax[0,idx].set_yticks([0,20,40,60,80,100,120,140,160,180])
#     ax[0,idx].tick_params(axis='both', which='both', pad=tick_pad)
#     ax[0,idx].set_title(zone)
#     ax[0,idx].annotate('$R^2$ = {}'.format(round(R2,2)), xy=(0.7, 0.9), xycoords='axes fraction', fontsize=40)

#     sns.regplot(x=daily_average_LMP['{}_LMP'.format(zone)], y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[1,idx], color='royalblue', ci=None)
#     ax[1,idx].set_xlabel("Simulated LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     if idx == 0:
#         ax[1,idx].set_ylabel("Historical LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
#     else:
#         ax[1,idx].set_ylabel('')
#     ax[1,idx].tick_params(axis='both', which='both', pad=tick_pad)
    
# #creating a custom legend for the figure
# labels = []
# handles = []

# legend1 = Line2D([], [], color='black', linewidth=3)
# handles.insert(0,legend1)
# labels.insert(0,'Simulated')

# legend2 = Line2D([], [], color='crimson', linewidth=3)
# handles.insert(1,legend2)
# labels.insert(1,'Historical')

# plt.suptitle(sim_LMP_name[20:-4]+'_wght_avg_gen_cap_demand', fontsize=45, ha='center', weight = 'bold')
# fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.16, 0.75), frameon=True,fontsize=35)
# plt.tight_layout()
# plt.savefig('LMP_Plots/LMP_{}_wght_avg_gen_cap_demand.png'.format(sim_LMP_name[20:-4]), dpi=100, bbox_inches='tight')
# plt.show()
# plt.clf()

# return None



# LMP_validation('Model_results/duals_100_coal_25.csv')    
# LMP_validation('Model_results/duals_100_simple_25.csv')   

 
        
        
        
        
        
        
        
    
    
    
    



