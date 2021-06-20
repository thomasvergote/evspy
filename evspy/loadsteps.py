import pandas as pd

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