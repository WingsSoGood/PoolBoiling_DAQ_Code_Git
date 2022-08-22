import nidaqmx
import numpy as np
import collections
import pandas as pd
import os
from sys import exit
import matplotlib.pyplot as plt
import scipy as sp

def tc_cvs_lin(tc_pl = 1, x = 1): #tc_pl is the physical label of a thrmcpl
    A0,A1,A2,A3,A4,A5 = 0,0,0,0,0,0

    if tc_pl == 'TC_pl5':
        #TC6
        x = x*1e3
        A0 = 0
        A1 = 24.94302
        A2 = -0.16242
        A3 = 0.0207
        A4 = -7.62629E-4
    elif tc_pl == 'TC_pl11':
        A0 = 14.71776
        A1 = 20065.79863
    elif tc_pl == 'TC_pl18': #18
        A0 = 14.68921
        A1 = 20113.70163
    elif tc_pl == 'TC_pl12':#in ['TC_pl12','TC_pl18','TC_plB2']: #12
        A0 = 1.2593*0.1
        A1 = 2.4924e-4
        A2 = -4.3451e-5
        A3 = 7.4661e6
    elif tc_pl == 'TC_plB1' or tc_pl == 'TC_pl20':
        A0 = 14.64239
        A1 = 20066.85559
    elif tc_pl == 'TC_plB2': #B2
        A0 = 14.56388
        A1 = 20132.51028
    else:
        return
    y = A0 + A1*x + A2*x**2 + A3*x**3 + A4*x**4 + A5*x**5
    return y

def LoadCalibrationTables(tc_pls):
    tables = {}
    for key, value in tc_pls.items():
        tab_i = pd.read_csv('TC Re-calibration/'+value[-2:]+'.csv')
        x, y = tab_i['U'], tab_i['T']
        tables[value] = x, y
    return tables

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

def T_Surface_k(qb,T0,T_bulk,k=0):
    f = 0.0254
    D0, D1, D2 = f*.75, f*.938, f*.75
    q0 = qb
    q1 = q0*(D0/D1)**2
    T1 = T0
    x1, x2 = f*.096, f*.158
    T12 = T1 - x1/k*q1
    Ts = T12
    SH = T12 - T_bulk
    htc = qb/SH/1e3
    return Ts, SH, htc

class heat_calc():
    def __init__(self, t = 0, data = 0, chns = 0, tc_pls = 0, T_calib_lookup_tables =0):
        self.time = t
        self.OverHeat = 0
        k = 400 #W/mK
        x1 = 5.08e-3 #3.4925e-3 #m #strictly 3.5mm #5.08e-3
        for m, (key, value) in enumerate(tc_pls.items()):
            V = data[m][0]
            T_temp = lookupT(value, V, T_calib_lookup_tables)
            if key == 'T_block1':
#                self.V_block1 = V   
                T1 = T_temp
            elif key == 'T_block2':
#                self.V_block2 = V
                T2 = T_temp
            elif key == 'T_f':
#                self.V_block3 = V
                Tf = T_temp
                #putting block temperatures into a list and sort out later by their magnitude, because the actual order may be different in every run
            elif key == 'T_surf':
#                self.V_surf = V
                T0 = T_temp
            elif key == 'T_air':
                Tair = T_temp
            else:
                continue
                        
        self.T_bulk = Tf
        self.T0 = T0
        self.T1 = T1
        self.T2 = T2
        self.T_air = Tair
        dT_dx2 = (T1-T2)/(x1)#(3*T1 - 4*T2 + T3)/(2*x1)
        qb = -k*dT_dx2 #W/m2
        self.qb = qb/100**2 #W/cm2
        if qb >= 0:
            T_s, SH, htc = T_Surface_k(qb,T0,Tf,k=400)
        else:
            T_s = -1
            SH = -1
            htc = -1
        self.T_s = T_s
        self.SH = SH
        self.htc = htc
        self.RegData = 0

def heat_shutdown():
    with nidaqmx.Task() as task:
        task.do_channels.add_do_chan(lines='cDAQ1Mod3/port0/line0')
        task.do_channels.add_do_chan(lines='cDAQ1Mod3/port0/line1')
        task.do_channels.add_do_chan(lines='cDAQ1Mod3/port0/line2')
        task.do_channels.add_do_chan(lines='cDAQ1Mod3/port0/line3')
        task.do_channels.add_do_chan(lines='cDAQ1Mod4/port0/line0')
        task.do_channels.add_do_chan(lines='cDAQ1Mod4/port0/line1')
        task.do_channels.add_do_chan(lines='cDAQ1Mod4/port0/line2')
        task.do_channels.add_do_chan(lines='cDAQ1Mod4/port0/line3')
        task.write(8*[False])   
    return

def heat_turnon():
    with nidaqmx.Task() as task:
        task.do_channels.add_do_chan(lines='cDAQ1Mod3/port0/line0')
        task.do_channels.add_do_chan(lines='cDAQ1Mod3/port0/line1')
        task.do_channels.add_do_chan(lines='cDAQ1Mod3/port0/line2')
        task.do_channels.add_do_chan(lines='cDAQ1Mod3/port0/line3')
        task.do_channels.add_do_chan(lines='cDAQ1Mod4/port0/line0')
        task.do_channels.add_do_chan(lines='cDAQ1Mod4/port0/line1')
        task.do_channels.add_do_chan(lines='cDAQ1Mod4/port0/line2')
        task.do_channels.add_do_chan(lines='cDAQ1Mod4/port0/line3')
        task.write(8*[True])   
    return

def Graphing(i=0, t=0, maresults=0, results=0, points_previous=0, xscale=0, iscale=0, fig = 0, ax1 = 0, ax2 = 0):
    qb = results.qb
    y = vars(results)

    if t < xscale:
        ax1.axes.set_xlim([0,xscale])
    else:
        ax1.axes.set_xlim([-xscale+t,t])
        
    pt = 8
    fs = 10
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.tick_params(axis='both', which='minor', labelsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    ax2.tick_params(axis='both', which='minor', labelsize=fs)
    ax1.grid(b=1)
    pT1 = ax1.scatter(t, y['SH'], c = 'r', s = pt, label = 'SH')
    pT5 = ax2.scatter(t, qb, c = 'c', s = pt, label = 'qb')
    if t == 0 :
        ax1.legend(loc='upper center', bbox_to_anchor=(0.3, -0.05),fancybox=True, shadow=True, ncol=5,fontsize=fs)
        ax2.legend(loc='upper center', bbox_to_anchor=(.9, -0.05),fancybox=True, shadow=True, ncol=5,fontsize=fs)
    if t > xscale:
        for px in points_previous[int(i%iscale)]:
            px.remove()
    fig.canvas.start_event_loop(0.133)
    fig.canvas.draw_idle()
    return [pT1,pT5]

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

def Overheat(T, T_limit):
    b = 0
    if (T > T_limit):
        b = 1
    return b

def buildQbArray(qb_stored,qb,std_window):
    if len(qb_stored) > std_window:
        qb_stored = np.roll(qb_stored,-1)
        qb_stored[-1] = qb

    else:
        qb_stored = np.append(qb_stored,qb)
  
    return qb_stored

def Steadiness(qb_stored,qb,std_window):
    if len(qb_stored) > std_window:
        std = np.std(qb_stored)
    else:
        std = 0
        
    return std

def qbTrend(qb_stored,qb):
    return qb-qb_stored[0]

def RegData(qb,l):
    qbr = round(qb,1)
    if qbr in l:
        index = l.index(qbr)
        l[index] = 999
    else:
        index = -1
    return index, l

def DisplayResults(results,SHc,qc,timec):
    for key, value in vars(results).items():
        print(key,':', round(value,1))
    print('SHc='+str(round(SHc)),'CHF='+str(round(qc)), timec)
    print('-')
    return

def defineChannels():
    chns = [('TC_pl20','cDAQ1Mod1/ai10'),
        ('TC_pl5','cDAQ1Mod2/ai5'),
        ('TC_plB1','cDAQ1Mod1/ai6'),
        ('TC_plB2','cDAQ1Mod1/ai7'),
        ('TC_pl11','cDAQ1Mod1/ai1'),
        ('TC_pl12','cDAQ1Mod1/ai2'),
        ('TC_pl18','cDAQ1Mod1/ai8'),
        ('TC_pl1','cDAQ1Mod2/ai1'),
        ('TC_pl3','cDAQ1Mod2/ai3'),
        ('TC_pl4','cDAQ1Mod2/ai4')]
    chns = collections.OrderedDict(chns)
    
    tc_pls = [('T_surf','TC_pl18'),
        ('T_block1','TC_pl12'), 
        ('T_block2','TC_plB1'),
        ('T_f','TC_plB2'),
        ('T_air','TC_pl20')]
    tc_pls = collections.OrderedDict(tc_pls)
    return chns, tc_pls

def checkDataFileOverwrite(filename):
    if os.path.exists(filename):
        exit('Old data file still there!')
    return

def main():
    i = 0
    t = 0
    xscale = 60 #seconds, it's the length of the x-axis on the plot
    sampling_rate = 1 #per second per channel
    iscale = xscale * sampling_rate
    points_previous = int(iscale)*[None]
    chns, tc_pls = defineChannels()    
    minv = -78e-3
    maxv = -minv
    df1 = pd.DataFrame()
    filename = 'Data.csv'
    checkDataFileOverwrite(filename)    
    f = open(filename, "w+")
    f.close()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    nidaqmx.constants.AutoZeroType = 10164
    T_calib_lookup_tables = LoadCalibrationTables(tc_pls)
    heat_turnon()
    
    with nidaqmx.Task() as task:
        for value in tc_pls.values():
            chn = chns[value]
            task.ai_channels.add_ai_voltage_chan(physical_channel=chn, min_val=minv, max_val=maxv)   
        task.timing.sample_mode='CONTINUOUS_SAMPLES'
        task.timing.samps_per_chan=1000
        task.timing.samp_clk_rate=sampling_rate
        data = task.read(number_of_samples_per_channel=1)
        qb_stored = np.array([])
        qc = 0
        SHc = 0
        timec = 0
        while i >= 0:
            data = task.read(number_of_samples_per_channel=1)
            results = heat_calc(t,data,chns,tc_pls,T_calib_lookup_tables)
            df1 = df1.append(pd.DataFrame([vars(results)]),ignore_index=True)
            std_window = 600
            qb_stored = buildQbArray(qb_stored,results.qb,std_window)
            qb_Trend = qbTrend(qb_stored,results.qb)
            if (t>0) and (t%(30) == 0):
                if os.stat(filename).st_size == 0:
                    df1.to_csv(filename, mode='a', header=True)
                else:
                    df1.to_csv(filename, mode='a', header=False)
                df1 = pd.DataFrame()   
            print('{}min Trend(qb)={}'.format(round(std_window/60),round(qb_Trend,3)))
            
            if dqb3_low(results.qb, qb_stored[-1], i, sampling_rate) or Overheat(results.T0, 200) or Overheat(results.T1, 300) or Overheat(results.T2, 300):
                heat_shutdown()
                results.OverHeat = 1
                if qc == 0 :
                    qc = results.qb
                    SHc = results.SH
                    timec = results.time

            DisplayResults(results,SHc,qc,timec)
            points_previous[int(i%iscale)] = Graphing(i=i,t=t,maresults=0,results=results,points_previous=points_previous,xscale=xscale,iscale=iscale,fig=fig,ax1=ax1,ax2=ax2) #store the current points and remove point fore-beyond the xscale
            i = i+1
            t = i/sampling_rate
      
    heat_shutdown()
    return

try:
    main()
except KeyboardInterrupt:
    heat_shutdown()
    print('Interrupted')
