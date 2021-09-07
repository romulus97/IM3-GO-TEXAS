# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 22:24:11 2021

@author: jkern
"""

# import libaries
import pandas as pd
import numpy as np
import networkx as nx
import mpl_toolkits
mpl_toolkits.__path__.append('/usr/lib/python3.7/dist-packages/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# load data

df_full = pd.read_csv('ERCOT_bus.csv',header=0)
df_selected = pd.read_csv('nodes_100.csv',header=0)
df_branches = pd.read_csv('branches_100.csv',header=0)
df_branches = df_branches[['fbus','tbus','rateA']]

buses = list(df_selected['bus_i'])

for b in buses:
    if buses.index(b) < 1:
        df_selected_GPS = df_full.loc[df_full['Number']==b,:]
    else:
        a = df_full.loc[df_full['Number']==b,:]
        df_selected_GPS = pd.concat([df_selected_GPS,a])
        
df_selected_GPS = df_selected_GPS.reset_index(drop=True)

graph = nx.from_pandas_edgelist(df_branches, 'fbus','tbus')

# Plot it
nx.draw(graph, with_labels=True)
plt.show()

plt.figure(figsize = (10,9))
m = Basemap(
    projection='merc',
    llcrnrlon=-180,
    llcrnrlat=10,
    urcrnrlon=-50,
    urcrnrlat=70,
    lat_ts=0,
    resolution='l',
    suppress_ticks=True)


mx, my = m(df_selected_GPS['Substation Longitude'].values, df_selected_GPS['Substation Latitude'].values)
pos = {}
for count, elem in enumerate (df_selected_GPS['Number']):
    pos[elem] = (mx[count], my[count])

nx.draw_networkx_nodes(G = graph, pos = pos, node_list = graph.nodes(), 
                        node_color = 'orange', alpha = 0.8, node_size = 3)
nx.draw_networkx_edges(G = graph, pos = pos, edge_color='red',
                        alpha=0.2, arrows = False)

m.drawcountries(linewidth = 1)
m.drawstates(linewidth = 0.2)
m.drawcoastlines(linewidth=1)
plt.tight_layout()
plt.savefig("map_1.png", format = "png", dpi = 300)
plt.show()

