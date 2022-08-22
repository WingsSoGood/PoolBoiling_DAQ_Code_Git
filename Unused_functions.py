# -*- coding: utf-8 -*-
"""
Unused functions

Created on Sat Nov 13 12:51:42 2021

@author: zhang367
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

def lookupT(tc, U, T_calib_lookup_tables):
    x, y = T_calib_lookup_tables[tc]
    f = sp.interpolate.interp1d(x,y,kind='quadratic',fill_value='extrapolate')
    T = float(f(U))
    return T

def qb3_lingress(k, x1, T1, T2, T3):
    x = np.array([0, x1, 2*x1])
    y = np.array([T1, T2, T3])    
    slope = (np.mean(x)*np.mean(y) - np.mean(x*y))/(np.mean(x)*np.mean(x) - np.mean(x*x))
    qb3 = k*slope
    return qb3

def qb3_Kandlikar(k,x1,T1,T2,T3):
    dT_dx3 = (3*T1 - 4*T2 + T3)/(2*x1)
    qb3 = -k*dT_dx3
    return qb3

def AddValueToCSV(filename,position,value):
    in_file = open("{}.csv".format(filename), "rb")
    reader = csv.reader(in_file)
    out_file = open("{}.csv".format(filename), "wb")
    writer = csv.writer(out_file)
    for row in reader:
        row[-1] = 4
        writer.writerow(row)
    in_file.close()    
    out_file.close()
    return

def LoadCalibrationTables(tc_pls):
    tables = {}
    Thermocouple_calibration_dictionary = {}
    for key, value in tc_pls.items():
        tab_i = pd.read_csv('TC Re-calibration/'+value[-2:]+'.csv')
        x, y = tab_i['U'], tab_i['T']
        Thermocouple_calibration_dictionary_i = dict(zip(tab_i['Rational43_p_name'], tab_i['Rational43_p_value']))
        tables[value] = x, y
        Thermocouple_calibration_dictionary[value] = Thermocouple_calibration_dictionary_i
    return tables, Thermocouple_calibration_dictionary

def htc3_low1(i=0,interval=0,filename=0,sampling_rate=0,results=0,htc3_checker=0):
    b = 0
    t = int(60*interval)
    if i > t*sampling_rate:
        df_prex =  pd.read_csv(filename, usecols=['htc3'], skiprows=range(1,i-t*sampling_rate)) #read the last x seconds of data
        htc_min = df_prex['htc3'].min()
        if results.htc3 < htc_min:
            b = 1
    return b

class htc3_low2():
    def __init__(self, filename=0, i=0, interval=0, results=0,sampling_rate=0, df_prex=0):
        self.df_prex = 0
        self.htc3_min = -1e6
        self.shutdown = 0
        t = int(60*interval)
        i_interval = int(t*sampling_rate)
        if i == i_interval:
            df_prex =  pd.read_csv(filename, usecols=['htc3'])#, skiprows=range(1,i-t*sampling_rate))
            self.df_prex = df_prex
            htc3_min = df_prex['htc3'].min()
            self.htc3_min = htc3_min

        elif i > i_interval:
            if results.htc3 < df_prex['htc3'].min():
                self.shutdown = 1
            df_prex = df_prex.drop(df_prex.index[[0]])
            df2 = pd.DataFrame([results.htc3], columns=['htc3'])
            #df_prex[-1] = results.htc3
            df_prex = df_prex.append(df2, ignore_index=True)
            self.df_prex = df_prex
            htc3_min = df_prex['htc3'].min()
            self.htc3_min = htc3_min

def dhtc3_low(Rst=0, Rst_pre=0, i=0, sampling_rate=0):
    b = 0
    if i >= 60*sampling_rate:
        thres = -2
        dhtc = Rst.htc3 - Rst_pre.htc3
        if dhtc < thres:
            b = 1
    return b

def dqb3_low(qb, qb_pre, i, sampling_rate):
    b = 0
    if i >= 60*sampling_rate:
        dq = qb - qb_pre
        if (qb>20) and (dq<-1):#-0.25:
            b = 1
    return b