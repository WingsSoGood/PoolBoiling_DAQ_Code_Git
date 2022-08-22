# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:53:43 2021

@author: wuxif
"""

import numpy as np
# import collections
# from pprint import pprint
# import sys
import pandas as pd
# import os
# from sys import exit
# import matplotlib.pyplot as plt
# import copy
# import scipy as sp
# from scipy import interpolate
# from scipy import stats
# from sklearn.linear_model import LinearRegression
# from decimal import Decimal

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

# Correct_qb_Measurement()
Correct_SH_Calculation()