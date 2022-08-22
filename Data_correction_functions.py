# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:53:43 2021

@author: wuxif
"""

import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
import os
from statistics import mean
import copy

def Correct_SH_Calculation():
    for i in range(182,200):
        try:
            df = pd.read_csv("{}.csv".format(i))
        except:
            print(i,'Not found')
            continue
        k = 15
        f = 0.0254
        D0, D1, D2 = f*.75, f*.938, f*.75
        q0 = df['qb'] * 1e4
        q1 = q0*(D0/D1)**2
        q2 = q0*(D0/D2)**2
        T1 = df['T0']
        x1, x2 = f*.096, f*.158
        T_roi = T1 - x1/k*q1 - x2/2/k*q2
        df['T_s'] = T_roi
        df['SH'] = T_roi - df['T_bulk']
        df['htc'] = df['qb']/df['SH']*10
        df.to_csv('{}-crt.csv'.format(i), index=False)
        # Ts = T12 - x2/k*q0
    return

def Mark_common_Tair_data():
    for i in range(220,221):
        try:
            df = pd.read_csv("{}.csv".format(i))
        except:
            print(i,'Not found')
            continue
        df['Usable mark'] = 0
        T_air_mode = df['T_air'].median()
        print(T_air_mode)
        df['Usable mark'].iloc[ (df['T_air'].subtract(T_air_mode).abs()) < 0.5] = 1
        # print((df['T_air'].subtract(T_air_mode).abs()))
        df.to_csv('{}-crt.csv'.format(i), index=False)
        # Ts = T12 - x2/k*q0
    return    

def f_heatflux_compensation(T_air, T1, T2, parameters):
    a, b =  parameters
    x = (T1+T2)/2 - T_air
    y = -a*x**b
    return y    

def HTC_Tair_compensation(parameters):
    for i in range(220,221):
        try:
            df = pd.read_csv("{}.csv".format(i))
        except:
            print(i,'Not found')
            continue
        k = 15
        f = 0.0254
        D0, D1, D2 = f*.75, f*.938, f*.75
        q0 = df['qb'] * 1e4
        
        f = f_heatflux_compensation(df['T_air'],df['T1'],df['T2'], parameters)
        q_crt = q0 + f
        
        q1 = q_crt*(D0/D1)**2
        q2 = q_crt*(D0/D2)**2
        T1 = df['T0']
        x1, x2 = f*.096, f*.158
        T_roi = T1 - x1/k*q1 - x2/2/k*q2
        
        df['qb_crt'] = q_crt/1e4
        df['T_s_crt'] = T_roi
        df['SH_crt'] = T_roi - df['T_bulk']
        df['htc_crt'] = df['qb']/df['SH_crt']*10
        df.to_csv('{}-crt.csv'.format(i), index=False)
        # Ts = T12 - x2/k*q0
    return    

def Correct_qb_Measurement():
    qb_lookup_table = pd.read_csv('qb_lookup_table_Average 137-138 subtract 118-120.csv',skiprows=[1])
    xp, fp = qb_lookup_table['qb_old'], qb_lookup_table['qb_new']
    for i in range(95,134):
        try:
            df = pd.read_csv("{}.csv".format(i))
        except:
            print(i) 
            continue
        df = df.rename(columns={"RegData": "qb_correct"})
        df['qb_correct'] = np.interp(df['qb'], xp, fp)
        df.to_csv('{}-crt.csv'.format(i), index=False)
    return    


def InsertCommentRow(df,label):
    df.loc[-1] = [label]*len(df.columns)  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index
    return df

def PermFilter(Ni,Nf):
    
    def Add_Result_to_dict(mydict, key, value):
        if key not in mydict.keys():
            mydict[key] = [value]
        else:
            mydict[key].append(value)
        return mydict
    
    def getAsymptoticHTC(dict_htc_asym, df,label,depth_dict):
        depth = depth_dict[label]
        if 'Q' in label:
            key_lv1 = 'GQD'
        elif 'M' in label:
            key_lv1 = 'G11D'
        else:
            key_lv1 = ''
            
        if depth < 1e-6: #this is the zero depth, baseline, give to both dictionary
            for key in dict_htc_asym.keys():
                # dict_htc_asym[key][depth] = df['htc'].min()
                dict_htc_asym[key]  = Add_Result_to_dict(dict_htc_asym[key], depth, df['htc'].min())
        
        else:
            if key_lv1 not in dict_htc_asym.keys():
                return dict_htc_asym
            
            dict_htc_asym[key_lv1] = Add_Result_to_dict(dict_htc_asym[key_lv1], depth, df['htc'].min())
        
        return dict_htc_asym
    
    def get_weight(myList):
        if len(myList) <= 1:
            return None
        myArray = np.array(myList)
        myMean = np.average(myArray)
        reciprocal_array = np.reciprocal(np.abs(myArray - myMean))
        weight = reciprocal_array/np.sum(reciprocal_array)  
        return weight        
        
    def calculate_AsymptoticHTC(mydict):
        newdict = copy.deepcopy(mydict)
        for key_lv1 in mydict.keys():
            for key in mydict[key_lv1].keys():
                newdict[key_lv1][key] = {}
                weight = get_weight(np.array(mydict[key_lv1][key]))
                # print(np.sum(weight))
                newdict[key_lv1][key]['Mean'] = np.average(np.array(mydict[key_lv1][key]),weights=weight)
                newdict[key_lv1][key]['Std'] = np.std(np.array(mydict[key_lv1][key]))
                # a = np.mean(np.array(mydict[key_lv1][key]))
        return newdict
        
    def Postprocess_HAD(htc_asym_dic,spacing_label):
        # df_htc_asym = pd.DataFrame(htc_asym_dic.items(), columns=['Depth', 'HTC After 20hr'])
        df_htc_asym = pd.DataFrame.from_dict(htc_asym_dic,orient='index')
        df_htc_asym = df_htc_asym.reset_index(level=0)
        df_htc_asym = df_htc_asym.rename(columns={'index':'Depth', 'Mean':'HTC After 20hr','Std':'HTC STD'})
        df_htc_asym = df_htc_asym.sort_values(by=['Depth'],ignore_index=1)    
        df_htc_asym = InsertCommentRow(df_htc_asym, spacing_label)
        df_htc_asym = df_htc_asym.reset_index()
        return df_htc_asym
    
    
    file_dict = {224:'M363_discard',
                 225:'Smooth',
                 226:'M211',
                 227:'M164',
                 230:'M262',
                 231:'M108_discard',
                 232:'Smooth_fractured',
                 233:'M211_discard',
                 234:'M363',
                 239:'Q100',
                 240:'Q100',
                 241:'Q210',
                 253:'Q360_discard',
                 254:'Q170',
                 255:'Q270',
                 256:'M164',
                 257:'M164',
                 258:'M108',
                 259:'M211_discard',
                 260:'Smooth',
                 261:'M262_discard',
                 262:'M363',
                 263:'M108',
                 264:'Q360',
                 265:'Q170',
                 266:'Q360',
                 267:'Q270_discard',
                 268:'Q210',
                 269:'Q100_discard',
                 270:'Q100_discard',
                 271:'M262', 
                 272:'Q100_discard',
                 273:'Q270',
                 274:'M211',
                 275:'M262'}
    
    
    depth_dict = {'Q40':42.5,
                  'Q100':102.6,
                  'Q170':165.4,
                  'Q210':205.3,
                  'Q270':271.8,
                  'Q320':319.3,
                  'Q360':361.8,
                  'Q430':429.3,
                  'M28':27.7,
                  'M69':69.2,
                  'M108':107.5,
                  'M164':164.2,
                  'M211':210.6,
                  'M262':262.4,
                  'M363':362.7,
                  'Smooth':0}
      
    df_master = pd.DataFrame(columns=['Time'])
    # htc_asym_GQD, htc_asym_G11D = {}, {}
    dict_htc_asym = {'GQD':{},
                     'G11D':{}}

    for i in range(Ni,Nf):
        if i not in file_dict.keys():
            continue
        
        if not os.path.isfile("{}.csv".format(i)):
            continue
        
        df = pd.read_csv("{}.csv".format(i))
        
        label = file_dict[i]
        if label not in depth_dict.keys():
            continue
        
        df = df[df['Data_Status']>0]
        df['Time'] = (df['time']-df['time'].min())/3600
        
        time_step = df['time'].iloc[1] - df['time'].iloc[0]        
        skip_points = int(60/time_step)
        df_sub = df[::skip_points][['T_bulk','T0','T1','T2','T_air','qb','T_s','SH','htc','Time']]
        
        df_sub['htc_sav_gol'] = uniform_filter1d(df_sub['htc'], size=11)
        
        # getAsymptoticHTC(df_sub[int(len(df_sub)/2):],label,depth_dict)
        # dict_htc_asym = getAsymptoticHTC(dict_htc_asym, df_sub[int(len(df_sub)/2):], label, depth_dict)
        dict_htc_asym = getAsymptoticHTC(dict_htc_asym, df_sub[df_sub['Time']>1.0], label, depth_dict)

        
        df_sub = InsertCommentRow(df_sub, file_dict[i])
        df_sub = df_sub.reset_index()
        df_sub.to_csv('{}-PermFilter.csv'.format(i), index=False)
        print("{}-{}.csv done".format(i,label))
        
        df_master = pd.concat([df_master,df_sub['htc_sav_gol']],axis=1)
        df_master['Time'] = df_sub['Time']
                
    # df_htc_asym_GQD = Postprocess_HAD(htc_asym_GQD,'Groove spacing = 5.00 mm')
    dict_htc_asym = calculate_AsymptoticHTC(dict_htc_asym)
    
    df_htc_asym_GQD = Postprocess_HAD(dict_htc_asym['GQD'],'Groove spacing = 5.00 mm')
    # df_htc_asym_G11D = Postprocess_HAD(htc_asym_G11D,'Groove spacing = 1.25 mm')
    df_htc_asym_G11D = Postprocess_HAD(dict_htc_asym['G11D'],'Groove spacing = 1.25 mm')
    
    df_htc_asym_GQD.to_csv('Compilation-HTCAsym-GQD.csv', index=False)
    df_htc_asym_G11D.to_csv('Compilation-HTCAsym-G11D.csv', index=False)
    df_master.to_csv('Compilation-Endurance.csv', index=False)


def BoilingCurve(Ni,Nf):
    file_dict = {238:'Smooth Pure',
                 242:'M211 Pure',
                 245:'M108 Pure',
                 246:'Q100 Pure',
                 247:'Q210 Pure',
                 248:'Q210 Salt',
                 249:'Q100 Salt',
                 250:'M108 Salt',
                 251:'Smooth Salt',
                 252:'M211 Salt'}
    
    df_master = pd.DataFrame()

    for i in range(Ni,Nf):
        if i not in file_dict.keys():
            # print('{} not in file_dict'.format(i))
            continue
        
        if not os.path.isfile("{}.csv".format(i)):
            continue
        
        df = pd.read_csv("{}.csv".format(i))
        
        label = file_dict[i]
        df['qb'] = df['qb']*10
        
        if i in [251, 252]:
            df['SH'] = df['SH']*1.15 #correct for the ice bath temperature shift
            df['qb'] = df['qb']*0.88
            df['htc'] = df['qb']/df['SH']
        # df['SH'] = df['SH']+ df['qb']*1e3*5e-3/15
        # df['htc'] = df['qb']/df['SH']
        df = df[df['Data_Status']>0]
        df = df.groupby(['Data_Status'], as_index=False).agg({'qb':['mean','std'],'SH':['mean','std'], 'htc':['mean','std']})
        df.columns = ['Data_Status','qb mean','qb std','SH mean','SH std','htc mean','htc std']
        
        df = InsertCommentRow(df, label.split(' ')[0])
        df = df.reset_index()
        df.to_csv('{}-{}-PermFilter.csv'.format(i,label), index=False)
        print("{}-{}.csv done".format(i,label))
        df_master = pd.concat([df_master,df],axis=1)

    df_master.to_csv('Compilation-BoilingCurves.csv', index=False)


# Correct_qb_Measurement()
# Correct_SH_Calculation()

# parameters = 0.001, 0.3
# HTC_Tair_compensation(parameters)

# Mark_common_Tair_data()

PermFilter(223,280)
BoilingCurve(237,260)