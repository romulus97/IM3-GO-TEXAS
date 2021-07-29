# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from geopy import distance
# import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
from matplotlib.colors import TwoSlopeNorm

# RTS = [150,125,100,75,50]
RTS = 75

df_BAs = pd.read_csv('BAs.csv',header=0)
BAs = list(df_BAs['Name'])

df = pd.read_csv('10k_load.csv',header=0)
crs = {'init':'epsg:4326'}
# crs = {"init": "epsg:2163"}
geometry = [Point(xy) for xy in zip(df['Substation Longitude'],df['Substation Latitude'])]
filter_nodes = gpd.GeoDataFrame(df,crs=crs,geometry=geometry)
nodes_df = gpd.GeoDataFrame(df,crs=crs,geometry=geometry)
nodes_df = nodes_df.to_crs(epsg=2163)

BAs_gdf = gpd.read_file('WECC.shp')
BAs_gdf = BAs_gdf.to_crs(epsg=2163)

states_gdf = gpd.read_file('geo_export_9ef76f60-e019-451c-be6b-5a879a5e7c07.shp')
states_gdf = states_gdf.to_crs(epsg=2163)

joined = gpd.sjoin(nodes_df,BAs_gdf,how='left',op='within')
joined2 = gpd.sjoin(nodes_df,states_gdf,how='left',op='within')
joined['State'] = joined2['state_name']

# buses = list(joined['Number'])
# B = []
# for b in buses:
#     if b in B:
#         pass
#     else:
#         B.append(b)

vmin, vmax, vcenter = 0, 500, 50
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = 'PiYG'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

r = 75

FN = 'duals100_LP.csv'
 
# selected nodes
df_prices = pd.read_csv(FN, header=0)
buses = list(df_prices['Bus'].unique())
selected = []

for b in buses:
    s = b[4:]
    selected.append(int(s))

selected_NODES = nodes_df[nodes_df['Number'].isin(selected)]
selected_NODES = selected_NODES.reset_index(drop=True)

    
    # locals()[FN + 'duals'] = np.zeros((8736,len(buses)))
    
    # avg_LMPs = []
    
    # for b in range(0,len(selected_NODES)):
        
    #     node = selected_NODES.loc[b,'Number']
    #     node2 = 'bus_' + str(node)
        
    #     sample = df_prices.loc[df_prices['Bus']==node2,'Value'].values
    #     # sample = sample.reset_index(drop=True)
        
    #     b_index = buses.index(node2)
    #     locals()[FN + 'duals'][:,b_index] = sample
    
    #     avg_LMPs.append(np.mean(sample))
        
    # selected_NODES['LMPs'] = avg_LMPs
    
    # # plt.figure()
    # # plt.plot(locals()[FN + 'duals'])
    # # plt.xlabel('Day of Year',fontweight = 'bold',fontsize=12)
    # # plt.ylabel('Shadow Price ($/MWh)',fontweight='bold',fontsize=12)
    # # plt.ylim([0,200])
    
    # # fig_name = 'Exp' + str(r) + '_coal.png'
    # # plt.savefig(fig_name,dpi=300)
    
    # fig,ax = plt.subplots()
    # states_gdf.plot(ax=ax,color='white',edgecolor='black',linewidth=0.5)
    # # nodes_df.plot(ax=ax,color = 'lightgray',alpha=1)
    # M=18
    
    # selected_NODES.plot(ax=ax , markersize= 100 ,cmap=cmap , norm=norm , column='LMPs' , marker='o' , edgecolor='black' ,linewidth=0.8 ,legend=True)
      
    # ax.set_box_aspect(1)
    # ax.set_xlim(-2000000,0)
    # ax.set_ylim([-1750000,750000])
    # plt.axis('off')
    # fn = 'map'+str(r) + '.jpg'
    # plt.savefig(fn,dpi=330)
    

