# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 20:32:00 2021

@author: kakdemi
"""

import pandas as pd
import numpy as np
import datetime
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from matplotlib.lines import Line2D

#preparing 2019 LMP data for MidC and Palo Verde (reading the data and interpolating the missing LMPs)
historical_LMP_2019 = pd.read_excel('ice_electric-2019final.xlsx', header=0,parse_dates=True,usecols=['Price hub', 'Trade date', 'Wtd avg price $/MWh'])
pricing_hubs = ['Mid C Peak','Palo Verde Peak']
dates_2019 = pd.date_range(start='1-1-2019',end='12-31-2019', freq='D')

for hub in pricing_hubs:
    
    filtered_LMPs = historical_LMP_2019.loc[historical_LMP_2019['Price hub']==hub].copy()
    del filtered_LMPs['Price hub']
    filtered_LMPs.set_index('Trade date',drop=True,inplace=True)
    filtered_LMPs = filtered_LMPs.reindex(dates_2019)
    filtered_LMPs_filled = filtered_LMPs.interpolate(method ='linear', limit_direction ='both')
    
    if hub == 'Mid C Peak':
        historical_LMP_2019_others_daily = filtered_LMPs_filled.copy()

    elif hub == 'Palo Verde Peak':
        historical_LMP_2019_others_daily = pd.concat([historical_LMP_2019_others_daily,filtered_LMPs_filled],axis=1)
        
historical_LMP_2019_others_daily.columns = ['MidC_LMP','PaloVerde_LMP']
    
    
#preparing 2019 LMP data for California (reading the data and finding daily LMPs)
historical_LMP_2019_CA = pd.read_csv('CAISO_data_2018_2020_RCA_Processed.csv', header=0,usecols=['datetime','PGAE_DAM_LMP','SCE_DAM_LMP','SDGE_DAM_LMP'])
historical_LMP_CA_dates = list(historical_LMP_2019_CA['datetime'])
historical_LMP_CA_dates = [pd.to_datetime(datetime.datetime.strptime(date, '%m-%d-%Y %H:%M')) for date in historical_LMP_CA_dates]
historical_LMP_2019_CA.index = historical_LMP_CA_dates
historical_LMP_2019_CA_daily = historical_LMP_2019_CA.resample('D').mean()
historical_LMP_2019_CA_daily.columns = ['PGAE_LMP', 'SCE_LMP', 'SDGE_LMP']
historical_LMP_all = pd.concat([historical_LMP_2019_CA_daily, historical_LMP_2019_others_daily], axis=1)


#reading bus to BA information and retail service territory information
bus_to_BA = pd.read_csv('nodes_to_BA_state.csv', header=0)
retail_service_areas = gpd.read_file('Electric-Retail-Service-Territories-shapefile/Retail_Service_Territories.shp')
retail_service_areas = retail_service_areas.to_crs(epsg=2163)

df_load_10k = pd.read_csv('10k_load.csv',header=0)
crs = {'init':'epsg:4326'}
geometry = [Point(xy) for xy in zip(df_load_10k['Substation Longitude'],df_load_10k['Substation Latitude'])]
nodes_load_10k = gpd.GeoDataFrame(df_load_10k,crs=crs,geometry=geometry)
nodes_load_10k = nodes_load_10k.to_crs(epsg=2163)


#defining MidC and PaloVerde BAs
MidC_BAs = ['PACIFICORP - WEST', 'PORTLAND GENERAL ELECTRIC COMPANY', 'PUGET SOUND ENERGY', 'SEATTLE CITY LIGHT', 'NORTHWESTERN ENERGY (NWMT)']
PaloVerde_BAs = ['ARIZONA PUBLIC SERVICE COMPANY', 'IDAHO POWER COMPANY', 'PACIFICORP - EAST', 'SALT RIVER PROJECT', 'PUBLIC SERVICE COMPANY OF NEW MEXICO']

#defining a function for LMP validation
def LMP_validation(sim_LMP_name):
    
    #copying necessary files defined globally
    bus_to_BA = globals()['bus_to_BA']
    MidC_BAs = globals()['MidC_BAs']
    PaloVerde_BAs = globals()['PaloVerde_BAs']
    nodes_load_10k = globals()['nodes_load_10k']
    retail_service_areas = globals()['retail_service_areas']
    historical_LMP_all = globals()['historical_LMP_all']
    df_load_10k = globals()['df_load_10k']
    
    #reading generation capacities at nodes
    gen_cap_node_all = pd.read_csv('Model_results/data_genparams{}.csv'.format(sim_LMP_name[19:-4]), header=0, usecols=['typ','node','maxcap'])
    gen_cap_node_all = gen_cap_node_all.loc[(gen_cap_node_all['typ']!='solar') & (gen_cap_node_all['typ']!='wind')]
    gen_cap_node_all = gen_cap_node_all.groupby('node').sum()

    #reading all buses and separating them with respct to their BAs or retail territories in WECC. 
    sim_LMP = pd.read_csv(sim_LMP_name, header=0)
    all_buses = list(sim_LMP['Bus'].unique())
    all_bus_numbers = [int(a[4:]) for a in all_buses]
    
    MidC_buses = []
    PaloVerde_buses = []
    PGAE_buses = []
    SCE_buses = []
    SDGE_buses = []
    
    for bus in all_bus_numbers:
        
        BA_specific_bus = bus_to_BA.loc[bus_to_BA['Number']==bus]['NAME'].values[0]
        
        if BA_specific_bus in MidC_BAs:
            MidC_buses.append('bus_{}'.format(bus))
            
        elif BA_specific_bus in PaloVerde_BAs:
            PaloVerde_buses.append('bus_{}'.format(bus))
        
        else: 
            bus_geometry = nodes_load_10k.loc[nodes_load_10k['Number']==bus]['geometry'].values[0]
            
            for index, cols in retail_service_areas.iterrows():
                
                if bus_geometry.within(cols['geometry']):
                    retail_service_name = cols['NAME']
                    
                    if retail_service_name == 'PACIFIC GAS & ELECTRIC CO.':
                        PGAE_buses.append('bus_{}'.format(bus))
                        
                    elif retail_service_name == 'SOUTHERN CALIFORNIA EDISON CO':
                        SCE_buses.append('bus_{}'.format(bus))
                        
                    elif retail_service_name == 'SAN DIEGO GAS & ELECTRIC CO':
                        SDGE_buses.append('bus_{}'.format(bus))
                        
                    else:
                        pass
                         
                else:
                    pass
                    
                    
    #trying different options to find the best fit
    
    #defining hours in 2019 and different LMP zones for validation
    hours_2019 = pd.date_range(start='1-1-2019 00:00:00',end='12-31-2019 23:00:00', freq='H')
    different_zones = ['MidC', 'PaloVerde', 'PGAE', 'SCE', 'SDGE']
    
    ######### 1st option: mean of all nodes within a zone #########
    
    #empty list to store R2s
    R2_mean_method = []
    
    #initilizing a figure
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.style.use('seaborn-whitegrid')
    
    axis_fontsize=33
    axis_label_pad = 10
    tick_pad = 10
    
    fig, ax = plt.subplots(2,5 , figsize=(60,20))
    
    #calculating R2 for every zone and plotting line and scatter plots to compare
    for zone in different_zones:
        
        zonal_prices = sim_LMP.loc[sim_LMP['Bus'].isin(locals()['{}_buses'.format(zone)])].copy()
        hourly_average_LMP = zonal_prices.groupby('Time').mean()
        hourly_average_LMP.index = hours_2019
        daily_average_LMP = hourly_average_LMP.resample('D').mean()
        daily_average_LMP.columns = ['{}_LMP'.format(zone)]
        
        daily_reg = linear_model.LinearRegression()
        daily_reg.fit(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
        R2 = daily_reg.score(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
        R2_mean_method.append(R2)
        idx = different_zones.index(zone)
        
        sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=daily_average_LMP['{}_LMP'.format(zone)], ax=ax[0,idx], color='black')
        sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[0,idx], color='crimson')
        ax[0,idx].set_xlabel("Date", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        if idx == 0:
            ax[0,idx].set_ylabel("LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        else:
            ax[0,idx].set_ylabel('')
        ax[0,idx].set_xticks(['2019-01-01', '2019-05-01', '2019-09-01', '2019-12-15']) 
        ax[0,idx].set_xticklabels(['Jan 2019', 'May 2019', 'Sep 2019', 'Dec 2019']) 
        # ax[0,idx].set_yticks([0,20,40,60,80,100,120,140,160,180])
        ax[0,idx].tick_params(axis='both', which='both', pad=tick_pad)
        ax[0,idx].set_title(zone)
        ax[0,idx].annotate('$R^2$ = {}'.format(round(R2,2)), xy=(0.7, 0.9), xycoords='axes fraction', fontsize=40)
    
        sns.regplot(x=daily_average_LMP['{}_LMP'.format(zone)], y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[1,idx], color='royalblue', ci=None)
        ax[1,idx].set_xlabel("Simulated LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        if idx == 0:
            ax[1,idx].set_ylabel("Historical LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        else:
            ax[1,idx].set_ylabel('')
        ax[1,idx].tick_params(axis='both', which='both', pad=tick_pad)
        
    #creating a custom legend for the figure
    labels = []
    handles = []
    
    legend1 = Line2D([], [], color='black', linewidth=3)
    handles.insert(0,legend1)
    labels.insert(0,'Simulated')
    
    legend2 = Line2D([], [], color='crimson', linewidth=3)
    handles.insert(1,legend2)
    labels.insert(1,'Historical')
    
    plt.suptitle(sim_LMP_name[20:-4]+'_simple_mean', fontsize=45, ha='center', weight = 'bold')
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.16, 0.75), frameon=True,fontsize=35)
    plt.tight_layout()
    plt.savefig('LMP_Plots/LMP_{}_basic_mean.png'.format(sim_LMP_name[20:-4]), dpi=100, bbox_inches='tight')
    plt.show()
    plt.clf()
        

    ######### 2nd option: weighted average with respect to demand at nodes #########
          
    #empty list to store R2s
    R2_weighted_average_demand = []
    
    #initilizing a figure
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.style.use('seaborn-whitegrid')
    
    axis_fontsize=33
    axis_label_pad = 10
    tick_pad = 10
    
    fig, ax = plt.subplots(2,5 , figsize=(60,20))
    
    #calculating R2 for every zone and plotting line and scatter plots to compare
    for zone in different_zones:
        
        wght_avg_demand_LMP_total = np.zeros((len(hours_2019)))
        buses_demand = []
        
        for bus in locals()['{}_buses'.format(zone)]:
            
            bus_number = int(bus[4:])
            zonal_prices = sim_LMP.loc[sim_LMP['Bus']==bus]['Value'].values
            df_load_10k.fillna(0,inplace=True)
            bus_demand = df_load_10k.loc[df_load_10k['Number']==bus_number]['Load MW'].values[0]
            if bus_demand == np.nan or bus_demand == 'nan' or bus_demand == '':
                bus_demand = 0
            buses_demand.append(bus_demand)
            wght_avg_demand_LMP_total = wght_avg_demand_LMP_total + zonal_prices*bus_demand
            
        wght_avg_demand_LMP_final = wght_avg_demand_LMP_total/sum(buses_demand)
        
        
        hourly_average_LMP = pd.DataFrame(wght_avg_demand_LMP_final)
        hourly_average_LMP.index = hours_2019
        daily_average_LMP = hourly_average_LMP.resample('D').mean()
        daily_average_LMP.columns = ['{}_LMP'.format(zone)]
        
        daily_reg = linear_model.LinearRegression()
        daily_reg.fit(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
        R2 = daily_reg.score(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
        R2_weighted_average_demand.append(R2)
        idx = different_zones.index(zone)
        
        sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=daily_average_LMP['{}_LMP'.format(zone)], ax=ax[0,idx], color='black')
        sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[0,idx], color='crimson')
        ax[0,idx].set_xlabel("Date", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        if idx == 0:
            ax[0,idx].set_ylabel("LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        else:
            ax[0,idx].set_ylabel('')
        ax[0,idx].set_xticks(['2019-01-01', '2019-05-01', '2019-09-01', '2019-12-15']) 
        ax[0,idx].set_xticklabels(['Jan 2019', 'May 2019', 'Sep 2019', 'Dec 2019']) 
        # ax[0,idx].set_yticks([0,20,40,60,80,100,120,140,160,180])
        ax[0,idx].tick_params(axis='both', which='both', pad=tick_pad)
        ax[0,idx].set_title(zone)
        ax[0,idx].annotate('$R^2$ = {}'.format(round(R2,2)), xy=(0.7, 0.9), xycoords='axes fraction', fontsize=40)
    
        sns.regplot(x=daily_average_LMP['{}_LMP'.format(zone)], y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[1,idx], color='royalblue', ci=None)
        ax[1,idx].set_xlabel("Simulated LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        if idx == 0:
            ax[1,idx].set_ylabel("Historical LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        else:
            ax[1,idx].set_ylabel('')
        ax[1,idx].tick_params(axis='both', which='both', pad=tick_pad)
        
    #creating a custom legend for the figure
    labels = []
    handles = []
    
    legend1 = Line2D([], [], color='black', linewidth=3)
    handles.insert(0,legend1)
    labels.insert(0,'Simulated')
    
    legend2 = Line2D([], [], color='crimson', linewidth=3)
    handles.insert(1,legend2)
    labels.insert(1,'Historical')
    
    plt.suptitle(sim_LMP_name[20:-4]+'_wght_avg_demand', fontsize=45, ha='center', weight = 'bold')
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.16, 0.75), frameon=True,fontsize=35)
    plt.tight_layout()
    plt.savefig('LMP_Plots/LMP_{}_wght_avg_demand.png'.format(sim_LMP_name[20:-4]), dpi=100, bbox_inches='tight')
    plt.show()
    plt.clf()
    
    
    ######### 3rd option: weighted average with respect to generation capacity at nodes #########
          
    #empty list to store R2s
    R2_weighted_average_gen = []
    
    #initilizing a figure
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.style.use('seaborn-whitegrid')
    
    axis_fontsize=33
    axis_label_pad = 10
    tick_pad = 10
    
    fig, ax = plt.subplots(2,5 , figsize=(60,20))
    
    #calculating R2 for every zone and plotting line and scatter plots to compare
    for zone in different_zones:
        
        wght_avg_gen_LMP_total = np.zeros((len(hours_2019)))
        buses_gen_cap = []
        
        for bus in locals()['{}_buses'.format(zone)]:
            
            bus_number = int(bus[4:])
            zonal_prices = sim_LMP.loc[sim_LMP['Bus']==bus]['Value'].values
    
            try:
                bus_gen = gen_cap_node_all.loc[bus,'maxcap']
            except KeyError:
                bus_gen = 0
                
            if bus_gen == np.nan or bus_gen == 'nan' or bus_gen == '':
                bus_gen = 0
            buses_gen_cap.append(bus_gen)
            wght_avg_gen_LMP_total = wght_avg_gen_LMP_total + zonal_prices*bus_gen
            
        wght_avg_gen_LMP_final = wght_avg_gen_LMP_total/sum(buses_gen_cap)
        
        
        hourly_average_LMP = pd.DataFrame(wght_avg_gen_LMP_final)
        hourly_average_LMP.index = hours_2019
        daily_average_LMP = hourly_average_LMP.resample('D').mean()
        daily_average_LMP.columns = ['{}_LMP'.format(zone)]
        
        daily_reg = linear_model.LinearRegression()
        daily_reg.fit(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
        R2 = daily_reg.score(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
        R2_weighted_average_gen.append(R2)
        idx = different_zones.index(zone)
        
        sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=daily_average_LMP['{}_LMP'.format(zone)], ax=ax[0,idx], color='black')
        sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[0,idx], color='crimson')
        ax[0,idx].set_xlabel("Date", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        if idx == 0:
            ax[0,idx].set_ylabel("LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        else:
            ax[0,idx].set_ylabel('')
        ax[0,idx].set_xticks(['2019-01-01', '2019-05-01', '2019-09-01', '2019-12-15']) 
        ax[0,idx].set_xticklabels(['Jan 2019', 'May 2019', 'Sep 2019', 'Dec 2019']) 
        # ax[0,idx].set_yticks([0,20,40,60,80,100,120,140,160,180])
        ax[0,idx].tick_params(axis='both', which='both', pad=tick_pad)
        ax[0,idx].set_title(zone)
        ax[0,idx].annotate('$R^2$ = {}'.format(round(R2,2)), xy=(0.7, 0.9), xycoords='axes fraction', fontsize=40)
    
        sns.regplot(x=daily_average_LMP['{}_LMP'.format(zone)], y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[1,idx], color='royalblue', ci=None)
        ax[1,idx].set_xlabel("Simulated LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        if idx == 0:
            ax[1,idx].set_ylabel("Historical LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        else:
            ax[1,idx].set_ylabel('')
        ax[1,idx].tick_params(axis='both', which='both', pad=tick_pad)
        
    #creating a custom legend for the figure
    labels = []
    handles = []
    
    legend1 = Line2D([], [], color='black', linewidth=3)
    handles.insert(0,legend1)
    labels.insert(0,'Simulated')
    
    legend2 = Line2D([], [], color='crimson', linewidth=3)
    handles.insert(1,legend2)
    labels.insert(1,'Historical')
    
    plt.suptitle(sim_LMP_name[20:-4]+'_wght_avg_gen_cap', fontsize=45, ha='center', weight = 'bold')
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.16, 0.75), frameon=True,fontsize=35)
    plt.tight_layout()
    plt.savefig('LMP_Plots/LMP_{}_wght_avg_gen_cap.png'.format(sim_LMP_name[20:-4]), dpi=100, bbox_inches='tight')
    plt.show()
    plt.clf()
    
    
    ######### 4th option: weighted average with respect to generation capacity + demand at nodes #########
          
    #empty list to store R2s
    R2_weighted_avg_sum_gen_demand = []
    
    #initilizing a figure
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.style.use('seaborn-whitegrid')
    
    axis_fontsize=33
    axis_label_pad = 10
    tick_pad = 10
    
    fig, ax = plt.subplots(2,5 , figsize=(60,20))
    
    #calculating R2 for every zone and plotting line and scatter plots to compare
    for zone in different_zones:
        
        wght_avg_gen_demand_LMP_total = np.zeros((len(hours_2019)))
        buses_gen_cap_demand = []
        
        for bus in locals()['{}_buses'.format(zone)]:
            
            bus_number = int(bus[4:])
            zonal_prices = sim_LMP.loc[sim_LMP['Bus']==bus]['Value'].values
    
            try:
                bus_gen = gen_cap_node_all.loc[bus,'maxcap']
            except KeyError:
                bus_gen = 0
                
            if bus_gen == np.nan or bus_gen == 'nan' or bus_gen == '':
                bus_gen = 0
                
            df_load_10k.fillna(0,inplace=True)
            bus_demand = df_load_10k.loc[df_load_10k['Number']==bus_number]['Load MW'].values[0]
            if bus_demand == np.nan or bus_demand == 'nan' or bus_demand == '':
                bus_demand = 0
            
            bus_info_weight = bus_demand + bus_gen
            buses_gen_cap_demand.append(bus_info_weight)
            wght_avg_gen_demand_LMP_total = wght_avg_gen_demand_LMP_total + zonal_prices*bus_info_weight
            
        wght_avg_gen_demand_LMP_final = wght_avg_gen_demand_LMP_total/sum(buses_gen_cap_demand)
        
        
        hourly_average_LMP = pd.DataFrame(wght_avg_gen_demand_LMP_final)
        hourly_average_LMP.index = hours_2019
        daily_average_LMP = hourly_average_LMP.resample('D').mean()
        daily_average_LMP.columns = ['{}_LMP'.format(zone)]
        
        daily_reg = linear_model.LinearRegression()
        daily_reg.fit(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
        R2 = daily_reg.score(daily_average_LMP['{}_LMP'.format(zone)].values.reshape(-1, 1), historical_LMP_all['{}_LMP'.format(zone)].values.reshape(-1, 1))
        R2_weighted_avg_sum_gen_demand.append(R2)
        idx = different_zones.index(zone)
        
        sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=daily_average_LMP['{}_LMP'.format(zone)], ax=ax[0,idx], color='black')
        sns.lineplot(x=daily_average_LMP['{}_LMP'.format(zone)].index, y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[0,idx], color='crimson')
        ax[0,idx].set_xlabel("Date", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        if idx == 0:
            ax[0,idx].set_ylabel("LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        else:
            ax[0,idx].set_ylabel('')
        ax[0,idx].set_xticks(['2019-01-01', '2019-05-01', '2019-09-01', '2019-12-15']) 
        ax[0,idx].set_xticklabels(['Jan 2019', 'May 2019', 'Sep 2019', 'Dec 2019']) 
        # ax[0,idx].set_yticks([0,20,40,60,80,100,120,140,160,180])
        ax[0,idx].tick_params(axis='both', which='both', pad=tick_pad)
        ax[0,idx].set_title(zone)
        ax[0,idx].annotate('$R^2$ = {}'.format(round(R2,2)), xy=(0.7, 0.9), xycoords='axes fraction', fontsize=40)
    
        sns.regplot(x=daily_average_LMP['{}_LMP'.format(zone)], y=historical_LMP_all['{}_LMP'.format(zone)], ax=ax[1,idx], color='royalblue', ci=None)
        ax[1,idx].set_xlabel("Simulated LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        if idx == 0:
            ax[1,idx].set_ylabel("Historical LMP ($/MWh)", labelpad=axis_label_pad, weight = 'bold', fontsize=axis_fontsize)
        else:
            ax[1,idx].set_ylabel('')
        ax[1,idx].tick_params(axis='both', which='both', pad=tick_pad)
        
    #creating a custom legend for the figure
    labels = []
    handles = []
    
    legend1 = Line2D([], [], color='black', linewidth=3)
    handles.insert(0,legend1)
    labels.insert(0,'Simulated')
    
    legend2 = Line2D([], [], color='crimson', linewidth=3)
    handles.insert(1,legend2)
    labels.insert(1,'Historical')
    
    plt.suptitle(sim_LMP_name[20:-4]+'_wght_avg_gen_cap_demand', fontsize=45, ha='center', weight = 'bold')
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.16, 0.75), frameon=True,fontsize=35)
    plt.tight_layout()
    plt.savefig('LMP_Plots/LMP_{}_wght_avg_gen_cap_demand.png'.format(sim_LMP_name[20:-4]), dpi=100, bbox_inches='tight')
    plt.show()
    plt.clf()
    
    return None



LMP_validation('Model_results/duals_100_coal_25.csv')    
LMP_validation('Model_results/duals_100_simple_25.csv')   

 
        
        
        
        
        
        
        
    
    
    
    



