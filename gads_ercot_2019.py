# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import sklearn
import math
#%matplotlib inline
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#Global variables
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
from math import sqrt
from statistics import mean
from statistics import median

import datetime

#import the data
#gads_events_19=pd.read_csv(r"C:\Users\hssemba\Documents\GADS_data\GADS_Events_2019_20220207_sv.csv" ,encoding='cp1252')
gads_events_19x=pd.read_csv(r"C:\Users\hssemba\Documents\GADS_data\GADS_Events_2019_20220207_imp.csv", encoding='cp1252')
gads_performance_19=pd.read_csv(r"C:\Users\hssemba\Documents\GADS_data\GADS_Performance_2019_20220208_sv.csv")
gads_units=pd.read_csv(r"C:\Users\hssemba\Documents\GADS_data\GADS_Units_20220207_sv.csv")

#import the 2018 file
gads_events_18x=pd.read_csv(r"C:\Users\hssemba\Documents\GADS_data\GADS_Events_2018_20220207_imp.csv", encoding='cp1252')


#manipulate the 2018 dataset
#we only want the events that end in 2019
#we want to remain with a subset of the 2018 dataset where the end date is 2019
gads_events_18x= gads_events_18x[gads_events_18x.EventEndDT.str.contains("2019") == True]


#change the start date to 2019 so because we are interested in counting duration from the start of 2019
gads_events_18x['EventStartDT']="1/1/2019 0:00"

#Adjusting the duration for 2018 to only retain the 2019 time period
start_time= datetime.datetime(2019, 1, 1, 00, 00, 00)

#end date to datetime format
gads_events_18x['EventEndDT']= pd.to_datetime(gads_events_18x['EventEndDT'], errors='coerce')

#end date to datetime format
list_1_2018=gads_events_18x['EventEndDT'].values.astype('datetime64[s]').tolist()
list_1_2018


#for the events from the previous year carrying over to this year, Duration is the difference between start time (beginning of year to end date)
hours_difference_18 = [abs(start_time - x).total_seconds() / 3600.0 for x in list_1_2018]
gads_events_18x["newDuration"]=hours_difference_18



#fill null values with zero for now
gads_events_19x['Duration'] = gads_events_19x['Duration'].fillna("0:")

#gads_ercot_19["newstartDT"]= gads_ercot_19["newstartDT"].split(sep, 1)[0]
#gads_events_19x["newDuration"] = pd.DataFrame([ x.split(':',1)[0] for x in gads_events_19x['Duration']])
gads_events_19x["newDuration"] = gads_events_19x['Duration'].str.split(':').str[0]
gads_events_19x["newDuration"] = gads_events_19x["newDuration"].astype(int)

#we want only events that start in 2019
#(I added this because gads_events_19 dataset has data that starts in 2020 and ends in 2020. We don't want that)
gads_events_19x= gads_events_19x[gads_events_19x.EventStartDT.str.contains("2019") == True]


#merge the gads_events_18x and gads_events_19x
gads_events_19=pd.concat([gads_events_18x, gads_events_19x])

#gads events start date to datetime
gads_events_19['EventStartDT']= pd.to_datetime(gads_events_19['EventStartDT'], errors='coerce')

#merge the units table and the events table
gads_df_19 = pd.merge(gads_events_19, gads_units,  how='left', on="UnitID")

#subsetting the TRE (ERCOT)
gads_ercot_19=gads_df_19[gads_df_19["RegionCode"]=="TRE"]

#...........................................................................................................................

#preprocessing
#1. Delete "Pumped Storage/Hydro", 'Miscellaneous', 'Internal Combustion/Reciprocating Engines','Geothermal'
gads_ercot_19=gads_ercot_19[((gads_ercot_19.UnitTypeCodeName != 'Pumped Storage/Hydro') &( gads_ercot_19.UnitTypeCodeName != 'Miscellaneous') & (gads_ercot_19.UnitTypeCodeName != 'Internal Combustion/Reciprocating Engines') & ( gads_ercot_19.UnitTypeCodeName != 'Geothermal'))]


#2. Replace with "Gas" :'Gas Turbine/Jet Engine (Simple Cycle Operation)','CC GT units', 'Combined Cycle Block', 'CC steam units', 'CoG GT units'
gads_ercot_19['UnitTypeCodeName'] = gads_ercot_19['UnitTypeCodeName'].replace(['Gas Turbine/Jet Engine (Simple Cycle Operation)','CC GT units ','Combined Cycle Block','CC steam units','CoG GT units'], 'Gas')

#3. Replace with "Coal" :'Fossil-Steam','Fluidized Bed','CoG steam units', 'Co-generator Block', 'Multi-boiler/Multi-turbine' 

gads_ercot_19['UnitTypeCodeName'] = gads_ercot_19['UnitTypeCodeName'].replace(['Fossil-Steam','Fluidized Bed','CoG steam units ','Co-generator Block ','Multi-boiler/Multi-turbine'],'Coal')



#.............................................................................................................................


#exclude 'Reserve Shutdown', 'Retired' from "EventTypeName"
gads_ercot_19=gads_ercot_19[((gads_ercot_19.EventTypeName != 'Reserve Shutdown') & ( gads_ercot_19.EventTypeName != 'Retired'))]

#.............................................................................................................................



#import ercot generators (from the ERCOT model (Dr. Kern))
ercot_gens=pd.read_csv(r"C:\Users\hssemba\Documents\GADS_data\ERCOT_Generators.csv")

#retain only natural gas, coal and nuclear
#(remove wind, solar, hydro
ercot_gens_2=ercot_gens[((ercot_gens.FuelType != 'WND (Wind)') & ( ercot_gens.FuelType != 'SUN (Solar)') & ( ercot_gens.FuelType != 'WAT (Water)'))]
 

#.................................................................................................................................

#categorizing the MWmax
a_0_49=[]
a_50_99=[]
a_100_199=[]
a_200_299=[]
a_300_399=[]
a_400_599=[]
a_600_799=[]
a_800_999=[]
a_1k_plus=[]

## "All" and ">200"
a_all_200=[]

for i in ercot_gens["MWMax"]:
    if i < 50:
        a_0_49.append(i)
    elif i < 100:
        a_50_99.append(i)
    elif i < 200:
        a_100_199.append(i)
    elif i < 300:
        a_200_299.append(i)
    elif i < 400:
        a_300_399.append(i)
    elif i < 600:
        a_400_599.append(i)
    elif i < 800:
        a_600_799.append(i)
    elif i < 1000:
        a_800_999.append(i)
    else:
        a_1k_plus.append(i)
        

#median of each category
md_0_49=median(a_0_49)
md_50_99=median(a_50_99)
md_100_199=median(a_100_199)
md_200_299=median(a_200_299)
md_300_399=median(a_300_399)
md_400_599=median(a_400_599)
md_600_799=median(a_600_799)
md_800_999=median(a_800_999)
md_1k_plus=median(a_1k_plus)


# "All" and ">200"
for i in ercot_gens["MWMax"]:
    if i >= 200:
        a_all_200.append(i)
#median for all gens with maxcap >=200         
md_all_200=median(a_all_200)     
md_all_200 


# 0-100 category
a_0_99=[]

for i in ercot_gens["MWMax"]:
    if i < 200:
        a_0_99.append(i)
md_0_99=median(a_0_99)        
       

       

#..................................................................................................................................
#adding a new column for new est cap
def categorise(row):  
    if row['RatingMW_BrochureGroup'] == "All" and row['RatingMW_grp'] == 3.0:
        return md_all_200
    elif row['RatingMW_BrochureGroup'] == "All" and row['RatingMW_grp'] == 2.0:
        return md_100_199
    elif row['RatingMW_BrochureGroup'] == "All" and row['RatingMW_grp'] == 1.0:
        return md_0_99
    elif row['RatingMW_BrochureGroup'] == "<50":
        return md_0_49
    elif row['RatingMW_BrochureGroup'] == "50+":
        return md_50_99
    elif row['RatingMW_BrochureGroup'] == "<100":
        return md_50_99
    elif row['RatingMW_BrochureGroup'] == "100-199":
        return md_100_199
    elif row['RatingMW_BrochureGroup'] == "200-299":
        return md_200_299
    elif row['RatingMW_BrochureGroup'] == "300-399":
        return md_300_399
    elif row['RatingMW_BrochureGroup'] == "400-599":
        return md_400_599
    elif row['RatingMW_BrochureGroup'] == "600-799":
        return md_600_799
    elif row['RatingMW_BrochureGroup'] == "800+":
        return md_800_999
    elif row['RatingMW_BrochureGroup'] == "1000+":
        return md_1k_plus

    #else:
       # return 0
    
#new size
gads_ercot_19['est_capacity'] = gads_ercot_19.apply(lambda row: categorise(row), axis=1)
#.................................................................................................................................

#adding a new column to identify the categories created
#n="and" "ovr"= over
def new_row_name(row):  
    if row['RatingMW_BrochureGroup'] == "All" and row['RatingMW_grp'] == 3.0:
        return "All_n_ovr_200"
    elif row['RatingMW_BrochureGroup'] == "All" and row['RatingMW_grp'] == 2.0:
        return "All_n_100_200"
    elif row['RatingMW_BrochureGroup'] == "All" and row['RatingMW_grp'] == 1.0:
        return "All_n_0_100"
    elif row['RatingMW_BrochureGroup'] == "<50":
        return "below_50"
    elif row['RatingMW_BrochureGroup'] == "50+":
        return "50_100"
    elif row['RatingMW_BrochureGroup'] == "<100":
        return "50_100"
    elif row['RatingMW_BrochureGroup'] == "100-199":
        return "100_200"
    elif row['RatingMW_BrochureGroup'] == "200-299":
        return "200_300"
    elif row['RatingMW_BrochureGroup'] == "300-399":
        return "300_400"
    elif row['RatingMW_BrochureGroup'] == "400-599":
        return "400_600"
    elif row['RatingMW_BrochureGroup'] == "600-799":
        return "600_800"
    elif row['RatingMW_BrochureGroup'] == "800+":
        return "800_1000"
    elif row['RatingMW_BrochureGroup'] == "1000+":
        return "ovr_1000"
    
gads_ercot_19['class_category'] = gads_ercot_19.apply(lambda row: new_row_name(row), axis=1)

#..................................................................................................................................


#in this section I change the format of the start date and the duration to one that I can use (hours 1-8760)



#converting time to a range of 1 to 8760
list_1_2019=gads_ercot_19['EventStartDT'].values.astype('datetime64[s]').tolist()
list_1_2019

#converting time to a range of 1 to 8760
hours_difference_2019 = [abs(start_time - x).total_seconds() / 3600.0 for x in list_1_2019]
gads_ercot_19["newstartDT"]=hours_difference_2019 

#also for all the events that started in the previous year the the "newstartST" is zero but we want it to be 1
gads_ercot_19["newstartDT"] = gads_ercot_19["newstartDT"].replace(0, 1)

#convert from float value to integer
gads_ercot_19["newstartDT"] = gads_ercot_19["newstartDT"].astype(int)



#new duration is integer
gads_ercot_19["newDuration"] = gads_ercot_19["newDuration"].astype(int)


#computing the unavailable capacity
#nac_perc fillna with zero
gads_ercot_19["newNAC_perc"]=gads_ercot_19["NAC_perc"].fillna(value=0)


#Unavailable cap
gads_ercot_19["lost_cap"]=(1- gads_ercot_19["newNAC_perc"])*gads_ercot_19["est_capacity"]




#subsetting the data to what we want to use
gads_ercot_19_2= gads_ercot_19[["EventID", "EventStartDT", "UnitTypeCodeName", "RatingMW_BrochureGroup","newstartDT" , "Duration", "newDuration" , "est_capacity", "lost_cap" ,"class_category"]]



#unique plant types
unique_unit_type=sorted(set(gads_ercot_19_2['UnitTypeCodeName']))

#unique class category
unique_class_category=sorted(set(gads_ercot_19_2['class_category']))

#list of all possible combinations of unique plant type and category
import random
random.seed(10)
list_of_combinations = [[type, cat] for type in unique_unit_type for cat in unique_class_category]


#new names for column to be created

new_col_name=[]
for unit_type, unit_class in (list_of_combinations):
    name=str(unit_type)+"_"+ str(unit_class)
    new_col_name.append(name)

#.........................................................................................................................
#code for unavailable capacity for every hour for every unit type and class categoty
new_cap=[]
for unit_type, unit_class in (list_of_combinations):
    
    df_subset = gads_ercot_19_2[(gads_ercot_19_2['UnitTypeCodeName'] == unit_type) & (gads_ercot_19_2['class_category'] == unit_class)]
    #print(df_subset)
    
    T=np.zeros((len(df_subset), 8770))
    

    start=list(df_subset["newstartDT"])
    dur=list(df_subset["newDuration"])
    cap=list(df_subset["lost_cap"])
  

  
    for i in range(0, len(df_subset)):
        
        start_pt=start[i]-1
        for j in range(0, dur[i]):
            T[i, start_pt + j]=cap[i]
    s=list(np.sum(T, axis=0))
    new_cap.append(s)  
    
        
    
#.............................................................................................................................

#putting all this in a new dataframe
df_new=pd.DataFrame(new_cap)
ercot_19_lostcap = df_new.transpose()
ercot_19_lostcap.columns=new_col_name
ercot_19_lostcap


#add a new column with hours from 1 to 8760
ercot_19_lostcap.insert(0, "Time", range(1, 1 + len(ercot_19_lostcap)))
ercot_19_lostcap


#cutting off everything greater than 8760
#keep only rows with time from 1 to 8760
ercot_19_lostcap=ercot_19_lostcap.iloc[0:8760, :]


#hourly_sum_lostcap
hourly_sum_lostcap=ercot_19_lostcap.sum(axis=1)

#sum of every column
hourly_sum_lostcap2=ercot_19_lostcap.sum(axis=0)


#importing the timestamp for the plotting
timestamp=pd.read_csv(r"C:\Users\hssemba\Documents\GADS_data\timestamp.csv")
timestamp['sced_timestamp']= pd.to_datetime(timestamp['sced_timestamp'], errors='coerce') 

#................................................................................................................
#converting to list
tNuclear_300_400 =ercot_19_lostcap.Nuclear_300_400.values.tolist()
tNuclear_600_800  =ercot_19_lostcap.Nuclear_600_800.values.tolist()
tNuclear_800_1000 =ercot_19_lostcap.Nuclear_800_1000.values.tolist()
tNuclear_400_600 =ercot_19_lostcap.Nuclear_400_600.values.tolist()
tNuclear_All_n_0_100=ercot_19_lostcap.Nuclear_All_n_0_100.values.tolist()
tNuclear_50_100 =ercot_19_lostcap.Nuclear_50_100.values.tolist()
tNuclear_All_n_ovr_200 =ercot_19_lostcap.Nuclear_All_n_ovr_200.values.tolist()
tNuclear_ovr_1000 =ercot_19_lostcap.Nuclear_ovr_1000.values.tolist()
tNuclear_100_200 =ercot_19_lostcap.Nuclear_100_200.values.tolist()
tNuclear_All_n_100_200 =ercot_19_lostcap.Nuclear_All_n_100_200.values.tolist()
tNuclear_below_50 =ercot_19_lostcap.Nuclear_below_50.values.tolist()
tNuclear_200_300 =ercot_19_lostcap.Nuclear_200_300.values.tolist()
tGas_300_400 =ercot_19_lostcap.Gas_300_400.values.tolist()
tGas_600_800 =ercot_19_lostcap.Gas_600_800.values.tolist()
tGas_800_1000 =ercot_19_lostcap.Gas_800_1000.values.tolist()
tGas_400_600 =ercot_19_lostcap.Gas_400_600.values.tolist()
tGas_All_n_0_100 =ercot_19_lostcap.Gas_All_n_0_100.values.tolist()
tGas_50_100 =ercot_19_lostcap.Gas_50_100.values.tolist()
tGas_All_n_ovr_200 =ercot_19_lostcap.Gas_All_n_ovr_200.values.tolist()
tGas_ovr_1000 =ercot_19_lostcap.Gas_ovr_1000.values.tolist()
tGas_100_200 =ercot_19_lostcap.Gas_100_200.values.tolist()
tGas_All_n_100_200 =ercot_19_lostcap.Gas_All_n_100_200.values.tolist()
tGas_below_50 =ercot_19_lostcap.Gas_below_50.values.tolist()
tGas_200_300 =ercot_19_lostcap.Gas_200_300 .values.tolist()
tCoal_300_400 =ercot_19_lostcap.Coal_300_400.values.tolist()
tCoal_600_800 =ercot_19_lostcap.Coal_600_800.values.tolist()
tCoal_800_1000 =ercot_19_lostcap.Coal_800_1000.values.tolist()
tCoal_400_600 =ercot_19_lostcap.Coal_400_600.values.tolist()
tCoal_All_n_0_100 =ercot_19_lostcap.Coal_All_n_0_100.values.tolist()
tCoal_50_100 =ercot_19_lostcap.Coal_50_100.values.tolist()
tCoal_All_n_ovr_200 =ercot_19_lostcap.Coal_All_n_ovr_200.values.tolist()
tCoal_ovr_1000 =ercot_19_lostcap.Coal_ovr_1000.values.tolist()
tCoal_100_200 =ercot_19_lostcap.Coal_100_200.values.tolist()
tCoal_All_n_100_200 =ercot_19_lostcap.Coal_All_n_100_200.values.tolist()
tCoal_below_50 =ercot_19_lostcap.Coal_below_50.values.tolist()
tCoal_200_300 =ercot_19_lostcap.Coal_200_300.values.tolist()

#....................................................................................................................
#reading data genparams
datagen=pd.read_csv(r"C:\Users\hssemba\Documents\GitHub\Exp100_simple_100\data_genparams.csv", encoding='cp1252')

#adding a new column to categorize the ranges
def gencat(row):  
    if row['maxcap'] <= 50 :
        return "below_50"
    elif row['maxcap'] <= 100:
        return "50_100"
    elif row['maxcap'] <= 200:
        return "100_200"
    elif row['maxcap'] <= 300:
        return "200_300"
    elif row['maxcap'] <= 400:
        return "300_400"
    elif row['maxcap'] <= 600:
        return "400_600"
    elif row['maxcap'] <= 800:
        return "600_800"
    elif row['maxcap'] <= 1000:
        return "800_1000"
    else:
        return "ovr_1000"

    
#new size
datagen['gen_categ'] = datagen.apply(lambda row: gencat(row), axis=1)



#...................................................................................................................
Nu_All_n_100_200=[]
Nu_600_800 =[]
Nu_200_300 =[]
Nu_400_600=[]
Nu_300_400=[]
Nu_800_1000=[]
Nu_All_n_0_100=[]
Nu_100_200 =[]
Nu_All_n_ovr_200 =[]
Nu_50_100 =[]
Nu_below_50 =[]
Nu_ovr_1000 =[]
Ga_All_n_100_200 =[]
Ga_600_800 =[]
Ga_200_300  =[]
Ga_400_600 =[]
Ga_300_400  =[]
Ga_800_1000 =[]
Ga_All_n_0_100 =[]
Ga_100_200 =[]
Ga_All_n_ovr_200 =[]
Ga_50_100  =[]
Ga_below_50  =[]
Ga_ovr_1000  =[]
Co_All_n_100_200 =[]
Co_600_800 =[]
Co_200_300 =[]
Co_400_600 =[]
Co_300_400 =[]
Co_800_1000 =[]
Co_All_n_0_100 =[]
Co_100_200  =[]
Co_All_n_ovr_200 =[]
Co_50_100  =[]
Co_below_50  =[]
Co_ovr_1000 =[]



#........................................................................................................
for i in range(0, len(datagen)):
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"gen_categ"]=="below_50" :
        Ga_below_50.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"gen_categ"]=="below_50" :
        Co_below_50.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"gen_categ"]=="50_100" :
        Ga_50_100.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"gen_categ"]=="50_100" :
        Co_50_100.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"gen_categ"]=="100_200" :
        Ga_100_200.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"gen_categ"]=="100_200" :
        Co_100_200.append(datagen.loc[i,"name"])        
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"gen_categ"]=="200_300" :
        Ga_200_300.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"gen_categ"]=="200_300" :
        Co_200_300.append(datagen.loc[i,"name"])         
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"gen_categ"]=="300_400" :
        Ga_300_400.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"gen_categ"]=="300_400" :
        Co_300_400.append(datagen.loc[i,"name"])          
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"gen_categ"]=="400_600" :
        Ga_400_600.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"gen_categ"]=="400_600" :
        Co_400_600.append(datagen.loc[i,"name"])     
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"gen_categ"]=="600_800" :
        Ga_600_800.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"gen_categ"]=="600_800" :
        Co_600_800.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"gen_categ"]=="800_1000" :
        Ga_800_1000.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"gen_categ"]=="800_1000" :
        Co_800_1000.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"gen_categ"]=="ovr_1000" :
        Ga_ovr_1000.append(datagen.loc[i,"name"])
    if datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"gen_categ"]=="ovr_1000" :
        Co_ovr_1000.append(datagen.loc[i,"name"])
        
    if datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"maxcap"]<=100 :
        Ga_All_n_0_100.append(datagen.loc[i,"name"])
    elif datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"maxcap"]<=100 :
        Co_All_n_0_100.append(datagen.loc[i,"name"])     
    elif datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"maxcap"]<=200 :
        Ga_All_n_100_200.append(datagen.loc[i,"name"])
    elif datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"maxcap"]<=200 :
        Co_All_n_100_200.append(datagen.loc[i,"name"])
    elif datagen.loc[i,"typ"]=="ngcc" and datagen.loc[i,"maxcap"]>200 :
        Ga_All_n_ovr_200.append(datagen.loc[i,"name"])
    elif datagen.loc[i,"typ"]=="coal" and datagen.loc[i,"maxcap"]>200 :
        Co_All_n_ovr_200.append(datagen.loc[i,"name"])
#.....................................................................................................

#Dictionary

Dict = {}
#Dict["Nuclear_below_50"]= Nu_below_50
#Dict["Nuclear_50_100"]= Nu_50_100
#Dict["Nuclear_100_200"]= Nu_100_200
#Dict["Nuclear_200_300"] = Nu_200_300
#Dict["Nuclear_300_400"] = Nu_300_400
#Dict["Nuclear_400_600"] = Nu_400_600
#Dict["Nuclear_600_800"] = Nu_600_800
#Dict["Nuclear_800_1000"] = Nu_800_1000
#Dict["Nuclear_ovr_1000"]= Nu_ovr_1000
#Dict["Nuclear_All_n_0_100"] = Nu_All_n_0_100
#Dict["Nuclear_All_n_100_200"] = Nu_All_n_100_200
#Dict["Nuclear_All_n_ovr_200"]= Nu_All_n_ovr_200
Dict["Gas_below_50"]= Ga_below_50
Dict["Gas_50_100"]= Ga_50_100
Dict["Gas_100_200"]= Ga_100_200
Dict["Gas_200_300"] = Ga_200_300
Dict["Gas_300_400"] = Ga_300_400
Dict["Gas_400_600"] = Ga_400_600
Dict["Gas_600_800"] = Ga_600_800
Dict["Gas_800_1000"] = Ga_800_1000
Dict["Gas_ovr_1000"]= Ga_ovr_1000
Dict["Gas_All_n_0_100"] = Ga_All_n_0_100
Dict["Gas_All_n_100_200"] = Ga_All_n_100_200
Dict["Gas_All_n_ovr_200"]= Ga_All_n_ovr_200
Dict["Coal_below_50"]= Co_below_50
Dict["Coal_50_100"]= Co_50_100
Dict["Coal_100_200"]= Co_100_200
Dict["Coal_200_300"] = Co_200_300
Dict["Coal_300_400"] = Co_300_400
Dict["Coal_400_600"] = Co_400_600
Dict["Coal_600_800"] = Co_600_800
Dict["Coal_800_1000"] = Co_800_1000
Dict["Coal_ovr_1000"]= Co_ovr_1000
Dict["Coal_All_n_0_100"] = Co_All_n_0_100
Dict["Coal_All_n_100_200"] = Co_All_n_100_200
Dict["Coal_All_n_ovr_200"]= Co_All_n_ovr_200



