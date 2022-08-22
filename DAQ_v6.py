import nidaqmx
import numpy as np
import collections
import pandas as pd
import sys, os
import traceback
from sys import exit
# import matplotlib.pyplot as plt
# import scipy as sp
from datetime import datetime


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

def LoadCalibrationFormula_Thermocouple(tc_pls):
    # tables = {}
    Thermocouple_calibration_dictionary = {}
    for key, value in tc_pls.items():
        tab_i = pd.read_csv('../TC Re-calibration/'+value[-2:]+'.csv')
        # x, y = tab_i['U'], tab_i['T']
        Thermocouple_calibration_dictionary_i = dict(zip(tab_i['Rational43_p_name'], tab_i['Rational43_p_value']))
        # tables[value] = x, y
        Thermocouple_calibration_dictionary[value] = Thermocouple_calibration_dictionary_i
    return Thermocouple_calibration_dictionary

def LoadCalibrationFormula_T1_T2():
    tab_i = pd.read_csv('../TC Re-calibration/T1_T2_Diff.csv')
    Thermocouple_calibration_dictionary_i = dict(zip(tab_i['Parameter'], tab_i['Value']))
    return Thermocouple_calibration_dictionary_i

def LoadCalibrationFormula_T_bulk():
    tab_i = pd.read_csv('../TC Re-calibration/T_bulk_Diff.csv')
    Thermocouple_calibration_dictionary_i = dict(zip(tab_i['Parameter'], tab_i['Value']))
    return Thermocouple_calibration_dictionary_i

def Rational43_conversion(label, U, Thermocouple_calibration_dictionary):
    if len(label) > 0:
        parameters = Thermocouple_calibration_dictionary[label]
    else:
        parameters = Thermocouple_calibration_dictionary
        
    p1 = parameters['p1']
    p2 = parameters['p2']
    p3 = parameters['p3']
    p4 = parameters['p4']
    q1 = parameters['q1']
    q2 = parameters['q2']
    q3 = parameters['q3']
    
    T_interpolation = (p1*U+p2*U**2+p3*U**3+p4*U**4)/(1+q1*U+q2*U**2+q3*U**3)
    return T_interpolation

def Rational5_conversion(label, U, Thermocouple_calibration_dictionary):
    if len(label) > 0:
        parameters = Thermocouple_calibration_dictionary[label]
    else:
        parameters = Thermocouple_calibration_dictionary
        
    a = parameters['a']
    b = parameters['b']
    c = parameters['c']
    d = parameters['d']
    
    T_interpolation = (a+b*U)/(1+c*U+d*U**2)
    return T_interpolation


def T_Surface_k(qb,T0,T_bulk,k_surface=0):
    f = 0.0254
    D0, D1, D2 = f*.75, f*.938, f*.75
    qb = qb*1e4 #convert from W/cm2 to W/m2
    q0 = qb
    q1 = q0*(D0/D1)**2
    q2 = q0*(D0/D2)**2
    x1, x2 = f*.096, f*.158
    Ts = T0 - x1/k_surface*q1 - x2/2/k_surface*q2
    SH = Ts - T_bulk
    htc = qb/SH/1e3 #convert from W/mK to kW/mK
    return Ts, SH, htc

class DataRecorder():
    def __init__(self,filename=0,results_class=0):
        self.df_save = pd.DataFrame()
        self.filename = filename
        self.results_class = results_class
        
    def Buffer_Store_data(self):
        self.df_save = self.df_save.append(pd.DataFrame([vars(self.results_class)]),ignore_index=True)
        if len(self.df_save.index) == 10:
            if os.stat(self.filename).st_size == 0:
                self.df_save.to_csv(self.filename, mode='a', header=True)
            else:
                self.df_save.to_csv(self.filename, mode='a', header=False)
            self.df_save = pd.DataFrame()


class Results():
    def __init__(self):
        self.time=0
        self.Overheat=0
        self.T_bulk = 0
        self.T0 = 0
        self.T1 = 0
        self.T2 = 0
        self.T_air = 0
        self.date_time = 0
        self.heater_status=0
        self.tc_pls = ''
        self.qb = 0
        self.T_s=0
        self.SH=0
        self.htc=0
        self.Data_Status=0
        
        
    def Update_data(self,t, data, chns, tc_pls, Thermocouple_calibration_dictionary, date_time):
        self.time = t
        self.date_time = date_time
        for m, (key, value) in enumerate(tc_pls.items()):
            V = data[m][0]
            # T_temp = lookupT(value, V, T_calib_lookup_tables)
            T_temp = Rational43_conversion(value, V, Thermocouple_calibration_dictionary)
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
        
    def Data_Reduction(self): 
        k = 400 #W/mK
        x1 = 5.08e-3 #3.4925e-3 #m #strictly 3.5mm #5.08e-3
        dT_dx2 = (self.T1-self.T2)/(x1)#(3*T1 - 4*T2 + T3)/(2*x1)
        self.qb = -k*dT_dx2 / 1e4 #W/cm2
        # self.qb = qb/100**2 #W/cm2
        if self.qb >= 0:
            self.T_s, self.SH, self.htc = T_Surface_k(self.qb,self.T0,self.T_bulk,k_surface=15)
        else:
            self.T_s = -1
            self.SH = -1
            self.htc = -1
        
    def getTC_configuration(self, tc_pls):
        self.tc_pls = tc_pls.items()
        return tc_pls.items()
    
    def Update_heater_status(self, HeaterControlClass):
        self.heater_status = HeaterControlClass.heater_status
    
    def Update_Data_Status(self, HeaterControlClass):
        self.Data_Status = HeaterControlClass.Data_Status    
    
    def Update_overheat_status(self,HeaterControlClass):
        self.Overheat = int(bool(len(HE_C.CHF_info.keys())))
    
    def Correct_T1T2_Difference(self, T1_T2_correction_dictionary):
        T1_T2_difference = Rational5_conversion('',self.T2, T1_T2_correction_dictionary)
        self.T2 = self.T2 + T1_T2_difference

    def Correct_Tbulk(self, Tbulk_correction_dictionary, Insu_Expose_Shfit=0):
        Tbulk_error = Rational5_conversion('',self.T_bulk, Tbulk_correction_dictionary) + Insu_Expose_Shfit
        self.T_bulk = self.T_bulk + Tbulk_error
    
    def Clear_TC_configuration(self):
        self.tc_pls = ''

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



def DisplayResults(results):
    for key, value in vars(results).items():
        if type(value) == float:
            print(key,':', round(value,2))
    # print('-')
    return
 
def F(n): #fibanacci sequence
    if n < 0: return 0
    if n == 0: return 0
    elif n == 1: return 1
    else: return F(n-1)+F(n-2)

class CameraControl():
    def __init__(self):
        self.Trigger_interval = 0
        # self.Trigger_interval_previous = 60*1
        self.Time_of_last_trigger = 0
        self.Trigger_count = 0
        # self.Sampling_rate = -1
        self.R = 0
        self.T_bulk_setpoint = 0
        # self.T_surface_setpoint = 0
        
    # def Update_Sampling_Rate(self,sampling_rate):
    #     self.Sampling_rate = sampling_rate
    
    def Update_Results(self, Results_class):
        self.R = Results_class
    def Update_T_bulk_setpoint(self, T_bulk_setpoint):
        self.T_bulk_setpoint = T_bulk_setpoint  
    
    # def Update_T_surface_setpoint(self, T_surface_setpoint):
    #     self.T_surface_setpoint = T_surface_setpoint      
    
    def Auto_Fibo_Trigger_interval(self):
        if self.Trigger_interval < 1800:
            # Trigger_interval_new = self.Trigger_interval + self.Trigger_interval_previous
            # self.Trigger_interval_previous = self.Trigger_interval
            # self.Trigger_interval = Trigger_interval_new
            self.Trigger_interval = F(self.Trigger_count-1)*60
        else:
            self.Trigger_interval = 1800
        
    def Trigger(self):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan('Dev1/ao1','relaychannel',0,5)
            task.write(0.0)
            task.write(5.0)
            task.write(0.0)
        self.Trigger_count += 1
        self.Time_of_last_trigger = self.R.time
         
    def Trigger_time_reached(self):
        print('NEXT TRIGGER IN {}'.format(self.Trigger_interval - (self.R.time - self.Time_of_last_trigger)))
        if self.R.time - self.Time_of_last_trigger >= self.Trigger_interval:
            return True
        else:
            return False
    
    def Auto_Trigger(self):
        if self.Trigger_count == 0:
            #capture initial surface
            self.Trigger()
        else:
            if self.R.qb>=2.5 and self.R.T1>130 and self.R.T_bulk>self.T_bulk_setpoint and self.Trigger_count == 1:
                #capture first time reaching qb=0
                self.Trigger()
            elif self.Trigger_count > 1 and self.Trigger_time_reached():
                self.Trigger()  
                self.Auto_Fibo_Trigger_interval()
                
        print('Trigger Count:',self.Trigger_count)


class HeaterControl():
    def __init__(self):
        self.R=0
        self.pre_qb = 0
        self.qb_trend = 0
        self.T0_trend = 0
        self.T_bulk_setpoint = 0
        self.T_bulk_Ready = 0
        self.T_surface_setpoint = 0
        self.CHF_info = {}
        self.heater_status = 0
        self.Aux_heater_status = 0
        self.Data_Status = 0
        self.time_start = 0
        self.isOverheat = 0
    
    def Update_T_bulk_setpoint(self, T_bulk_setpoint):
        self.T_bulk_setpoint = T_bulk_setpoint  
    
    def Update_T_surface_setpoint(self, T_surface_setpoint):
        self.T_surface_setpoint = T_surface_setpoint 
    
    def Update_Results(self,results=0, previous_qb=0, qb_trend=0, T0_trend=0):
        self.R=results
        self.pre_qb = previous_qb
        self.qb_trend = qb_trend
        self.T0_trend = T0_trend
        DisplayResults(results)
    
    def DisplayHeaterStatus(self):
        print('Cartridge_Heater_Status: {}'.format(self.heater_status))
        print('Aux_Heater_Status: {}'.format(self.Aux_heater_status))     
     
    def auxheater_switch(self,state):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan('Dev1/ao0','relaychannel',0,5)
            task.write(4.0*state)
        self.Aux_heater_status=state        

    
    def cartridge_heaters_switch(self, state):
        with nidaqmx.Task() as task:
            for i in range(3,5):
                for j in range(0,4):
                    task.do_channels.add_do_chan(lines='cDAQ1Mod{}/port0/line{}'.format(i,j))

            task.write(8*[bool(state)])
        self.heater_status=state        
    
    def qb_plummet(self):
        if self.R.time > 60 and self.R.qb < self.pre_qb-1:
            return True
        else:
            return False
    
    def Update_CHFinfo(self):
        if len(self.CHF_info.keys())==0:
            self.CHF_info = {'Status':'CHF_Triggered','SH_c':self.R.SH,'CHF':self.R.qb,'Time_c':self.R.time}
    
    def PreventOverHeat(self):
        if self.qb_plummet() or (self.R.T0 > 223) or any(y>300 for y in [self.R.T1, self.R.T2]):
            self.cartridge_heaters_switch(0)
            self.Update_CHFinfo()
            self.isOverheat = 1
            
    def PreventOverCool(self):
        if any(y<110 for y in [self.R.T1, self.R.T2]):
            self.cartridge_heaters_switch(1)
        
    def DisplayCHFinfo(self):
        for key, value in self.CHF_info.items():
            print(key,':',value)
    
    def Keep_T_x_at_setpoint(self,T_label):
        target = getattr(self.R,T_label)
        
        if T_label == 'T_bulk':
            set_point = self.T_bulk_setpoint
            control = self.auxheater_switch
        elif T_label == 'T0':
            set_point = self.T_surface_setpoint
            control =  self.cartridge_heaters_switch
        else:
            print('set point of {} not defined'.format(T_label))
            return
        
        if target < set_point:
            control(1)
        else:
            control(0)
    
    def Keep_qb_at_setpoint(self,set_point):
        target = self.R.qb
        
        if target < set_point:
            self.cartridge_heaters_switch(1)
        else:
            self.cartridge_heaters_switch(0)    
    
    def Keep_surface_warm(self,T_warm=105):
        if self.R.T0 < T_warm:
            self.cartridge_heaters_switch(1)
        else:
            self.cartridge_heaters_switch(0)

    def Update_T_bulk_Readiness(self):
        if self.R.T_bulk > self.T_bulk_setpoint-2:
            self.T_bulk_Ready = 1
            #once T_bulk is ready you do not want to reverse it
    
    def One_Time_Set_Timer(self):
        if self.time_start < 0.1:
            self.time_start = self.R.time
    
    def Reset_Timer(self):
        self.time_start = self.R.time
    
    def Change_Data_Status_to(self,status_input):
        self.Data_Status = status_input
    
    def All_Heaters_Ready(self):
        return (self.T_bulk_Ready and self.R.qb>=0 and self.R.T1>130)
    
    def Test_has_lasted_x_hours(self,period):
        if self.R.time - self.time_start >= period*3600:
            return True
        else:
            return False
    
    def Cycle(self,qb_gradient,T1_gradient):
        if (qb_gradient <= 0.07) and (T1_gradient <= 0.05) and (self.R.time>120):
            self.cartridge_heaters_switch(0)
    
    def AdiabaticCalibrationMode(self):
        self.PreventOverHeat()
        self.PreventOverCool()
    
    def KeepPoolTemperature(self):
        self.Keep_T_x_at_setpoint('T_bulk')
        self.Update_T_bulk_Readiness()
    
    def Initialize_by_condition(self,condition_func, set_timer_func):
        """
        

        Parameters
        ----------
        condition_func : TYPE
            The input funciton that defines the condition.
        *args : TYPE
            arguments of the input function.

        Returns
        -------
        Modifies attributes.

        """
        if condition_func():
        # if self.All_Heaters_Ready():
            # self.One_Time_Set_Timer()
            set_timer_func()
            # self.Change_Data_Status_to(1)    
    
    def Set_Mode(self, func, *args):
        self.KeepPoolTemperature()
        self.PreventOverHeat()
        if self.isOverheat:
            return
        
        if self.T_bulk_Ready:
            func(*args)
            # self.Keep_T_x_at_setpoint('T0')
        else:
            self.Keep_surface_warm()

    
    def Post_Dissolving_mode(self):
        self.Keep_T_x_at_setpoint('T_bulk')
        self.Keep_surface_warm()
        self.PreventOverHeat()         
    
    def Timed_Deposit_Dissolve(self,period,trigger_func,record_data_func):
        if self.Test_has_lasted_x_hours(period):
            self.Change_Data_Status_to(0)
            self.Post_Dissolving_mode()
        else:
            self.Set_Mode(self.Keep_T_x_at_setpoint,'T0')
            self.Initialize_by_condition(self.All_Heaters_Ready,self.One_Time_Set_Timer)
            if self.All_Heaters_Ready():
                self.Change_Data_Status_to(1)
        
        trigger_func()
        record_data_func()

    def Timed_Deposit_Mode(self,period,record_data_func):
        self.Set_Mode(self.Keep_T_x_at_setpoint,'T0')
        self.Initialize_by_condition(self.All_Heaters_Ready,self.One_Time_Set_Timer)
        if self.All_Heaters_Ready():
            self.Change_Data_Status_to(1)
        if self.Test_has_lasted_x_hours(period):
            self.Change_Data_Status_to(0)
        
        record_data_func()

    
    def BaselineEstablishMode(self,qb_Trend,T1_Trend,        record_data_func
):
        self.PreventOverHeat()
        self.Cycle(qb_Trend,T1_Trend)
        self.PreventOverCool()  
        record_data_func()
        
    def FreezeShut(self):
        self.cartridge_heaters_switch(0)
   
    def JustKeepWarm(self):
        self.Keep_T_x_at_setpoint('T_bulk')
        self.Keep_surface_warm()
        self.PreventOverHeat()       
        
    def T0_not_reaching(self):
        if self.R.T0 < self.T_surface_setpoint -1.5 or self.R.T0 > self.T_surface_setpoint + 1.5:
            return 1
    
    def T0_reaching(self):
        if self.R.T0 > self.T_surface_setpoint -0.1 and self.R.T0 < self.T_surface_setpoint + 0.1:
            return 1    
    
    def Timed_Boiling_Curve(self, trigger_func,         record_data_func, period=300/3600):
        
        if self.R.time < 1:
            self.T_surface_setpoint = 110
        
        self.Change_Data_Status_to(0)
        self.Set_Mode(self.Keep_T_x_at_setpoint,'T0')
        self.Initialize_by_condition(self.T0_not_reaching,self.Reset_Timer)
        
        if (self.R.time - self.time_start > 0.7*period*3600) and (self.R.time - self.time_start < period*3600):
            if self.T0_reaching():
                self.Change_Data_Status_to(self.T_surface_setpoint)
            
        if self.Test_has_lasted_x_hours(period):
            # self.Reset_Timer()
            self.T_surface_setpoint += 10
            trigger_func()

        print('T0_setpoint: {} C'.format(self.T_surface_setpoint))
        print('Time_lasted: {} s'.format(self.R.time - self.time_start))
        record_data_func()
      

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

def Trend(qb_stored,qb):
    # return qb-qb_stored[0]
    return np.std(qb_stored)

def RegData(qb,l):
    qbr = round(qb,1)
    if qbr in l:
        index = l.index(qbr)
        l[index] = 999
    else:
        index = -1
    return index, l

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
        
# =============================================================================
# def Buffer_Store_data(args):
#     df1, filename, results_class = args
#     
#     df1 = df1.append(pd.DataFrame([vars(results_class)]),ignore_index=True)
#     if len(df1.index) == 10:
#         if os.stat(filename).st_size == 0:
#             df1.to_csv(filename, mode='a', header=True)
#         else:
#             df1.to_csv(filename, mode='a', header=False)
#         df1 = pd.DataFrame()
#     return df1
# =============================================================================

def Average_data_of_the_last_X_seconds(x):
    return

def Record_data_every_X_seconds(x):
    return
    
def main(HE_C, RS_C, Cam_C, T_bulk_setpoint, T_surface_setpoint):
    i = 0
    t = 0
    # xscale = 60 #seconds, it's the length of the x-axis on the plot
    sampling_rate = 2 #per second per channel
    Cam_C.Update_T_bulk_setpoint(T_bulk_setpoint)
    HE_C.Update_T_bulk_setpoint(T_bulk_setpoint)
    HE_C.Update_T_surface_setpoint(T_surface_setpoint)
    chns, tc_pls = defineChannels()    
    minv = -78e-3
    maxv = -minv
    # df1 = pd.DataFrame()
    filename = '..\Data.csv'
    checkDataFileOverwrite(filename)    
    f = open(filename, "w+")
    f.close()
    
    Recorder_C = DataRecorder(filename = filename, results_class = RS_C)
    
    nidaqmx.constants.AutoZeroType = 10164
    
    Thermocouple_calibration_dictionary = LoadCalibrationFormula_Thermocouple(tc_pls)
    
    T1_T2_correction_dictionary = LoadCalibrationFormula_T1_T2()
    Tbulk_correction_dictionary = LoadCalibrationFormula_T_bulk()
    
    "**************************"
    with nidaqmx.Task() as task:
        for value in tc_pls.values():
            chn = chns[value]
            task.ai_channels.add_ai_voltage_chan(physical_channel=chn, min_val=minv, max_val=maxv)   
        task.timing.sample_mode='CONTINUOUS_SAMPLES'
        task.timing.samps_per_chan=1000
        task.timing.samp_clk_rate=sampling_rate
        data = task.read(number_of_samples_per_channel=1)
        qb_stored = np.array([])
        T0_stored = np.array([])
        
        while 1:
            data = task.read(number_of_samples_per_channel=1)
            RS_C.Update_data(t, data, chns, tc_pls, Thermocouple_calibration_dictionary, datetime.now())
            RS_C.Correct_T1T2_Difference(T1_T2_correction_dictionary)
            RS_C.Correct_Tbulk(Tbulk_correction_dictionary,Insu_Expose_Shfit=-0.42527)
            RS_C.Data_Reduction()
            
            if i==0:
                RS_C.getTC_configuration(tc_pls)
            else:
                RS_C.Clear_TC_configuration()
                
            std_window = 30
            qb_stored = buildQbArray(qb_stored,RS_C.qb,std_window)
            qb_Trend = Trend(qb_stored,RS_C.qb)
            print('Trend (Heat Flux):', qb_Trend)
            T0_stored = buildQbArray(T0_stored,RS_C.T0,std_window)
            T0_Trend = Trend(T0_stored,RS_C.T1) 
            print('Trend (T0):', T0_Trend)
            
            HE_C.Update_Results(RS_C,qb_stored[-1])
            
            Cam_C.Update_Results(RS_C)

            # HE_C.AdiabaticCalibrationMode()
            # HE_C.BaselineEstablishMode(qb_Trend,T1_Trend,Recorder_C.Buffer_Store_data)
            HE_C.Timed_Deposit_Dissolve(test_duration, Cam_C.Auto_Trigger, Recorder_C.Buffer_Store_data)
            # HE_C.Timed_Boiling_Curve(Cam_C.Trigger,Recorder_C.Buffer_Store_data)  
            # HE_C.KeepPoolTemperature()
            # HE_C.JustKeepWarm()
            
            RS_C.Update_overheat_status(HE_C)
            RS_C.Update_heater_status(HE_C)
            RS_C.Update_Data_Status(HE_C)
            HE_C.DisplayCHFinfo()
            
            # df1 = Buffer_Store_data(df1,filename,RS_C)
            
            HE_C.DisplayHeaterStatus()
            print('Data Status:',HE_C.Data_Status)
            i = i+1
            t = i/sampling_rate
            
            # Cam_C.Auto_Trigger()
            
            print('-')
    HE_C.cartridge_heaters_switch(0)
    return

HE_C = HeaterControl()
RS_C = Results()
Cam_C = CameraControl()
T_bulk_setpoint = 99
T_surface_setpoint = 180
test_duration = 20
try:
    main(HE_C,RS_C,Cam_C,T_bulk_setpoint,T_surface_setpoint)
except BaseException as e:
    HE_C.cartridge_heaters_switch(0)
    HE_C.auxheater_switch(0)
    print(traceback.format_exc())   