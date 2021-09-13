# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:36:51 2021

@author: kakdemi
"""

from glob import glob
import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# defining different model types  and a empty lists to store data
node_numbers = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
UC_treatment = ['simple','coal']
transmission_multiplier = [25, 50, 75, 100]

all_time_in_hrs = []
all_combinations = []

#saving current working directory
cwd = os.getcwd()

#creating an empty array to store runtimes in a tabular format
runtime_table = np.zeros((len(node_numbers),len(transmission_multiplier)*len(UC_treatment)))

#running the algorithm for each model type
for NN in node_numbers:
    
    for UC in UC_treatment:
       
        for TP in transmission_multiplier:
            
            #changing director and saving the current directory name (model type)
            directory_name = 'Exp{}_{}_{}'.format(NN,UC,TP)
            all_combinations.append(directory_name[3:])
            os.chdir('{}/{}'.format(cwd,directory_name))
            
            #checking if duals.csv is in the directory or not (if not, model run was not successful)
            if os.path.isfile('{}/{}/duals.csv'.format(cwd,directory_name)):
                
                #creating a list with the names of all files starting with the name 'out'
                out_files = glob('out*')
                file_sizes = []
                
                #calculating file sizes of all out files and storing them
                for out in out_files:
                    
                    with open(out, 'rt') as myfile:  

                        all_string = myfile.read()
                        file_sizes.append(int(sys.getsizeof(all_string)))
                
                #getting the index of the out file which takes the most space in memory
                index_of_max_file_size = file_sizes.index(max(file_sizes))
                
                #reading the out file which takes the most space in memory
                with open(out_files[index_of_max_file_size], 'rt') as myfile:
                    
                    all_string = myfile.read()
                    
                    #checking again if successful or not just in case
                    if "Successfully completed." in all_string:
                        
                        #saving runtimes in hrs and appending them
                        time = re.search("Run time :(.*?)sec.", all_string)
                        time = round(int(time.group(1))/3600,2)
                        all_time_in_hrs.append(time)
                                
                        if UC == 'simple':     
                            runtime_table[node_numbers.index(NN) ,transmission_multiplier.index(TP)] = time
                                    
                        elif UC == 'coal':      
                            runtime_table[node_numbers.index(NN) ,transmission_multiplier.index(TP)+4] = time
                                    
                        else:
                            pass
       
                    else:
                        pass
            
            #appending 100 hrs (a flag) for the failed simulations 
            else:
                all_time_in_hrs.append(100)
                        
                if UC == 'simple':              
                    runtime_table[node_numbers.index(NN) ,transmission_multiplier.index(TP)] = 100             
                
                elif UC == 'coal':             
                    runtime_table[node_numbers.index(NN) ,transmission_multiplier.index(TP)+4] = 100
            
            os.chdir(cwd)

#creating dataframes for runtimes and exporting those as an excel file            
Runtimes_df = pd.DataFrame(all_time_in_hrs, index=all_combinations, columns=['Runtime (hours)'])
Runtimes_df.to_excel('WECC_runtimes.xlsx', sheet_name='List')

Runtime_table_df = pd.DataFrame(data=runtime_table, index=node_numbers)
Runtime_table_df.columns = pd.MultiIndex.from_product([['LP (Simple)','MILP (Coal)'], transmission_multiplier])

with pd.ExcelWriter('WECC_runtimes.xlsx', engine='openpyxl', mode='a') as writer:  
    Runtime_table_df.to_excel(writer, sheet_name='Table')

#plotting a figure to compare runtimes
plt.rcParams.update({'font.size': 17})
plt.rcParams['font.sans-serif'] = "Arial"
plt.style.use('seaborn-whitegrid')
axis_fontsize=22
fig, ax = plt.subplots(figsize=(16,8))

Runtime_table_df.plot.bar(rot=0, ax=ax)
plt.legend(bbox_to_anchor=(0.5, -0.28), frameon=True,ncol=4, bbox_transform=ax.transAxes, loc='lower center')
ax.set_ylabel("Runtime (hours)", weight='bold', fontsize=axis_fontsize)
ax.set_xlabel("Number of nodes", weight='bold', fontsize=axis_fontsize)
ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100])

plt.savefig('Runtimes_comparison.png', dpi=250, bbox_inches='tight')
plt.show()
plt.clf()


