import pandas as pd
import numpy as np


class LoadSteps:
    def __init__(self):
        self.load_steps = [0]
        self.duration = [0]
        self.start_time = [0]
        self.end_time = [0]
        self.load = [0]
        self.type = ['IL']
        self.rate = [0]
        self.cv = [1]
        self.loaddf = pd.DataFrame(columns=['Load'])
    def add_load_step(self, duration, load, rate, type_test,cv):
        self.load_steps.append(len(self.load_steps))
        self.duration.append(duration)
        self.start_time.append(self.end_time[-1])
        self.end_time.append(self.start_time[-1]+duration)
        self.load.append(load)
        self.rate.append(rate)
        self.type.append(type_test)
        self.cv.append(cv)
        self.loaddf.loc[self.start_time[-1]]=self.load[-1]
        self.loaddf.loc[self.end_time[-1]-0.1]=self.load[-1]
    def change_load_step(self,load_step,duration,load, rate, type_test):
        self.duration[load_step]=duration
        self.load[load_step]=load
        self.rate[load_step]=rate
        self.type[load_step]=type_test
        self.end_time[load_step]=self.start_time[load_step]+duration
        self.loaddf = pd.DataFrame(columns=['Load'])
        for i in range(1,self.load_steps[-1]+1):
            self.start_time[i]=self.end_time[i-1]
            self.end_time[i]=self.start_time[i]+self.duration[i]
            self.loaddf.loc[self.start_time[i]]=self.load[i]
            self.loaddf.loc[self.end_time[i]-0.1]=self.load[i]
    def delete_load_step(self,load_step):
        self.load_steps.remove(load_step)
        self.duration.remove(self.duration[load_step])
        self.load.remove(self.load[load_step])
        for i in range(load_step,self.load_steps[-1]):
            self.load_steps[i]=self.load_steps[i-1]+1
            self.start_time[i]=self.end_time[i-1]
            self.end_time[i]=self.start_time[i]+self.duration[i]
        self.loaddf = pd.DataFrame(columns=['Load'])
        for i in range(1,self.load_steps[-1]+1):
            self.loaddf.loc[self.start_time[i]]=self.load[i]
            self.loaddf.loc[self.end_time[i]-0.1]=self.load[i]  

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def log_mean(x,y):
    return np.mean(np.interp(10**np.linspace(np.log10(100),np.log10(np.max(x)),100),x,y))

def identify_all_stages(oed_data):
    stages = list(set(oed_data['step_id']))
    stage_info={}
    for i in stages:
        stage_info[i]={}
        stage_data=oed_data[oed_data['step_id']==i]
        stage_info[i]['avg_void_ratio']=log_mean(stage_data['time'],stage_data['void_ratio'])
        stage_info[i]['avg_load']=np.average(stage_data['load'])
        stage_info[i]['initial_void_ratio']=stage_data['void_ratio'].iloc[0]
        stage_info[i]['final_void_ratio']=stage_data['void_ratio'].iloc[-1]
        stage_info[i]['sample_height']=stage_data['sample_height'].iloc[0]
        stage_info[i]['duration']=stage_data['time'].iloc[-1]-stage_data['time'].iloc[0]
        if i==1:
            if np.std(stage_data['void_ratio_rate'].iloc[5:-5])/np.nanmean(stage_data['void_ratio_rate'])<0.1:
                stage_info[i]['type']='CRS'
                stage_info[i]['strain_rate']=np.nanmedian(stage_data['void_ratio_rate'])
            else:
                stage_info[i]['type']='IL'
                stage_info[i]['strain_rate']=np.nan
        elif np.std(stage_data['void_ratio'])<1e-5:
            stage_info[i]['type']='R'
            stage_info[i]['strain_rate']=0
        elif np.abs(oed_data[oed_data['step_id']==i-1]['load'].iloc[-1]-stage_data['load'].iloc[-1])<0.1:
            stage_info[i]['type']='CL'
            stage_info[i]['strain_rate']=np.nan
        elif  np.abs((np.nanmean(stage_data['void_ratio_rate'].iloc[np.int(len(stage_data)/2):-1])-np.nanmean(stage_data['void_ratio_rate'].iloc[1:-np.int(len(stage_data)/2)]))/np.nanmean(stage_data['void_ratio_rate'].iloc[1:-1]))<0.05:
            stage_info[i]['type']='CRS'
            #print(np.nanmean(stage_data['void_ratio_rate'].iloc[int(len(stage_data)/3):int(len(stage_data)*2/3)]))
            stage_info[i]['strain_rate']=np.nanmean(stage_data['void_ratio_rate'].iloc[int(len(stage_data)/4):int(len(stage_data)*3/4)])
        else:
            stage_info[i]['type']='IL'
            stage_info[i]['strain_rate']=np.nan
        stage_info[i]['initial_OCR']=np.max(oed_data[oed_data['step_id']<i]['load'])/stage_data['load'].iloc[0]#oed_data[oed_data['step_id']==np.clip(i-1,1,np.infty)]['load'].iloc[-1]
        stage_info[i]['final_OCR']=np.max(oed_data[oed_data['step_id']<i]['load'])/stage_data['load'].iloc[-1]
        stage_info[i]['initial_load']=stage_data['load'].iloc[0]
        stage_info[i]['final_load']=stage_data['load'].iloc[-1]
        stage_info[i]['load_change']=stage_data['load'].iloc[-1]-stage_data['load'].iloc[0]
    return stage_info