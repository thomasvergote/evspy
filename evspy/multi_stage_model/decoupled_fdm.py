import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc
import tqdm
import logging

from evspy.multi_stage_model.consolidation import Consol_Terzaghi_Uavg_vertical
from evspy.multi_stage_model.viscous_swelling import swelling_law
from evspy.multi_stage_model.distorted_isotaches import base_isotache_model, create_isotache, make_contour

class distorted_isotache_model(base_isotache_model):
    '''Distorted isotache model with decoupled model 

    Parameters
    ----------
    base_isotache_model : class
        Base class for isotache models
    '''
    def initialize_decoupled(self):
        self.time_step_start={}
        self.time_step_end={}
        self.time=np.array([0])
        self.sigma=np.array([self.sigma0])
        self.e[:]=self.e_init
        self.initialize_reference_curve()
        self.sigpref[0] = self.get_sigpref(sigma=self.sigma[0:1],e=self.e[0:1],sigpref=[0],initiate=True)
        if self.sigpref[0]<self.sigma0:
            self.e[0] = self.eNC[self.sigrangeNC>self.sigma0][1]
            logging.warning(f'Initial effective stress is too high for given initial void ratio; reducing to einit = {self.e[0]}')
            self.sigpref[0] = self.get_sigpref(self.sigma[0:1],self.e[0:1],sigpref=[0],initate=True)
        self.OCR_real[0]=self.sigpref[0]/self.sigma[0]
        self.OCR[0]=self.sigpref[0]/self.sigma[0]
        #print(sigma[0],e[0],beta2,beta3,Cc,Cr,CalphaNC,sigpref[0],e0,isotache,power_law,ref_func)
        self.eratec[0]=self.calc_pos_isotache(self.sigma[0],self.e[0],self.sigpref[0])
        self.erates[0]=self.erateinits
        self.erate[0]=self.eratec[0]-self.erates[0]
        self.dt=1e-20#np.clip(1/np.abs(erate[0])*dtfactor,1e-5,1e-3)
        self.Calphahatc_real=np.zeros((self.dimt))
        self.Calphahats_real=np.zeros((self.dimt))
        self.OCRrate=np.zeros((self.dimt))
        self.Calphahats_real[0]=self.CalphaCc*self.Cc**10**self.b1*(self.OCR[0]-1)**self.m1
        
        #for t in tqdm.tqdm(range(1,dimt)):

    def initiate_new_load_step(self,load_step,t,running,reset_minimum_swelling_rate):
        next_load_step=False
        if self.load.type[load_step]=='IL':
            if (t>=len(self.time)-1): #(time[t]>=load.end_time[load_step])|
                #print(load_step)
                next_load_step=True
        elif self.load.type[load_step]=='CRS':
            if self.load.rate[load_step]>0:
                if self.sigma[t-1]>=self.load.load[load_step]:
                    next_load_step=True
            elif self.load.rate[load_step]<0:
                if self.sigma[t-1]<=self.load.load[load_step]:
                    next_load_step=True
            else:
                if (self.time[t]-self.time_step_end[self.load.type[load_step-1]+str(load_step-1)]>=self.load.duration[load_step]):
                    next_load_step=True
        if next_load_step:
            reset_minimum_swelling_rate = 0
            if len(self.load.type)>load_step+1:
                #print('next step '+str(load_step))
                self.time_step_end[self.load.type[load_step]+str(load_step)]=self.time[t]
                load_step+=1
                if self.load.type[load_step]=='IL':
                    time_u,U=Consol_Terzaghi_Uavg_vertical(self.load.cv[load_step]/3600/24/365,self.H,targettime=self.load.duration[load_step],dimt=1000,constant_dt_time=1e6,dtmax=1e5)
                    U=np.append(U[0],np.clip(U[1:]-U[:-1],0,1).cumsum())
                    self.time=np.append(self.time,self.time[-1]+0.01+time_u)
                    self.sigma=np.append(self.sigma,self.sigma[-1]+1/U[-1]*(self.load.load[load_step]-self.sigma[-1])*np.clip(U,0,1))
                elif self.load.type[load_step]=='CRS':
                    self.erate_crs = self.load.rate[load_step]
                self.time_step_start[self.load.type[load_step]+str(load_step)]=self.time[t]
            else:
                running=False
        return next_load_step, running, load_step, reset_minimum_swelling_rate
         
    def run_iterations(self,timefactor=1.2):
        t = 0
        #start=False
        i = 0
        failsafe=0
        dtmult = 1.01
        targettime=1e7
        dtfactor=1
        dtfactorinit=dtfactor
        OCR_ref=self.OCR_real[0]
        self.sigp_update[0]=self.sigpref[0]
        #while (t<dimt-1) and (dt>0): #(time[t]<targettime) and 
        printing=False
        load_step = 0
        next_load_step=True
        running=True
        reset_minimum_swelling_rate = 0
        while (running)&(t<self.dimt-1):
            next_load_step, running, load_step, reset_minimum_swelling_rate=self.initiate_new_load_step(load_step,t,running, reset_minimum_swelling_rate)
            if running:
                t+=1
            if self.load.type[load_step]=='CRS':
                if next_load_step:
                    dt = 1e-10
                else:
                    dt = np.clip(np.abs(1/self.erate_e[t-1])/15000,0.001,np.min([dt*timefactor,5e4]))
                self.time=np.append(self.time,[self.time[t-1]+dt])
                self.erate_e[t] = self.erate_crs+self.eratec[t-1]+self.erates[t-1]
                self.sigma=np.append(self.sigma,[self.sigma[t-1]+((10**(self.erate_e[t]*dt/self.Cr)-1)/dt)*self.sigma[t-1]*dt])
                self.ee[t]=self.ee[t-1]-self.erate_e[t]*dt 
            elif self.load.type[load_step]=='IL':
                dt=self.time[t]-self.time[t-1]
                self.erate_e[t]= self.Cr*0.434*(self.sigma[t]-self.sigma[t-1])/dt/np.average([self.sigma[t],self.sigma[t-1]])
                self.ee[t]=self.ee[t-1]-self.erate_e[t]*dt 
            
            self.OCR_real[t]=self.sigpref[t-1]/self.sigma[t]
            self.OCR[t]=np.clip(np.max(np.append(np.array([self.sigpref[0]]),self.sigma[:t]))/self.sigma[t],1.005,np.infty)
            self.OCRrate[t] = (self.OCR[t]-self.OCR[t-1])/dt
            
            sigmarate=(self.sigma[t-1]-self.sigma[t-2])/dt
            self.erates[t],self.Calphahats_real[t-1],reset_minimum_swelling_rate=swelling_law(self.erates[t-1],self.OCR[t],self.OCRrate[t],
                                                                        self.Calphahats_real[t-1],dt,self.CalphaCc*self.Cc,
                                                                        self.m1,self.b1,self.m2,self.b2,sigmarate,-self.erate_e[t],reset_minimum_swelling_rate)
            
            # Distortion occurs only during swelling and not during relaxation
            if ((self.sigma[t]<self.sigma[t-1]/1.000001)
                    &(self.sigma[t-1]<self.sigma[t-2]/1.000001)
                    &(np.abs(self.e[t]-self.e[t-1])>0)
                    &(np.abs(self.e[t-2]-self.e[t-1])>0)):
                if (self.load.type[load_step]=='IL')&(np.round(self.load.load[load_step-1],1)==np.round(self.load.load[load_step],1)):
                    self.sigp_update[t]=self.sigp_update[t-1]
                else:
                    self.sigp_update[t]=np.max(self.sigpref[self.sigpref>0])
            else:
                self.sigp_update[t]=self.sigp_update[t-1]

            self.e[t]=self.e[t-1]+(self.eratec[t-1]+self.erates[t-1])*dt+(self.ee[t]-self.ee[t-1])
            self.sigpref[t] = self.get_sigpref(self.sigma[t-1:t+1],self.e[t-1:t+1],self.sigpref[t-1:t])
            self.eratec[t]=self.calc_pos_isotache(self.sigma[t],self.e[t],self.sigp_update[t],steps=20)
            self.eratec[t]=-self.eratec[t]
            
            if np.isnan(self.ee[t]):
                print(dt)
                print(self.Cr*0.434*(self.sigma[t]-self.sigma[t-1])/dt/np.average([self.sigma[t],self.sigma[t-1]]))
                print(str(self.sigma[t-1])+' '+str(self.sigma[t]))
                print('failure')
            if self.eratec[t]>1:
                print((self.sigma[t],self.e[t],self.beta2, self.beta3,self.Cc,self.Cr,self.CalphaNC,self.sigp_update[t],self.e0))
            self.erate[t]=(self.e[t]-self.e[t-1])/dt#
            
            self.Calphahats_real[t]=self.CalphaCc*self.Cc*10**self.b1*(np.clip(self.OCR[t]-1,0.005,np.infty))**self.m1

        self.time = self.time[:t]
        self.sigma = self.sigma[:t]
        self.e = self.e[:t]
        self.erate = self.erate[:t]
        self.Calphahats_real = self.Calphahats_real[:t]
        self.sigpref = self.sigpref[:t]
        self.eratec = self.eratec[:t]
    
    def plotting_stresspath(self,sigp):
        dforg,eUC_org,eUC0_org,eNC_org,sigrange_org,sigrangeNC=create_isotache(erateref = self.erateref, beta2 = self.beta2, sigp=sigp,
                                                                                beta3 = self.beta3,Cc = self.Cc,Cr = self.Cr,CalphaNC = self.CalphaNC,
                                                                                e0=self.e0,isotache=self.isotache, power_law=self.use_power_law, 
                                                                                ref_func=self.ref_func, dsig=self.dsig, beta_nash=self.use_beta_nash,param_nash=self.param_nash)
        dforg=dforg.reset_index()
        cs=make_contour(dforg,eUC_org,eUC0_org,eNC_org,sigrange_org,sigrangeNC,num=20,figsize=(8,5),vmax=-3,colorbar=True)
        plt.plot(self.sigma[0],self.e[0],'o')
        plt.plot(self.sigma,self.e,'k-',lw=4)
        plt.ylim(np.min(self.e),np.max(self.e))
        plt.xlim(1,1000)
        #df,eUC,eUC0,eNC,sigrange,sigrangeNC=create_isotache(beta2 = beta2, beta3 = beta3,Cc = Cc,Cr = Cr,CalphaNC = CalphaNC,sigp=sigpref[t-1],e0=e0,isotache=isotache,power_law=power_law,ref_func=ref_func,dsig=0.05,beta_nash=beta_nash)
        #cs=make_contour(df,eUC,eUC0,eNC,sigrange,sigrangeNC,num=20,figsize=(8,5),vmax=-3)
        #plt.plot(sigma[0],e[0],'o')
        #plt.plot(sigma[:t],e[:t],lw=4)
        #plt.ylim(1,2.2)
        #plt.xlim(1,500)
    #print(load_step)
    #return time[:t],sigma[:t],e[:t],erate[:t],erate_e[:t],eratec[:t],erates[:t],Calphahats_real[:t],ee[:t],sigpref[:t],time_step,OCR[:t],OCR_real[:t],sigp_update[:t]

    def plotting_timeseries(self,loadstep : str, set_zero=False,**kws):
        time = self.time[(self.time>self.time_step_start[loadstep])&(self.time<self.time_step_end[loadstep])]-self.time[(self.time>=self.time_step_start[loadstep])][0]
        e = self.e[(self.time>self.time_step_start[loadstep])&(self.time<self.time_step_end[loadstep])]
        sigma = self.sigma[(self.time>self.time_step_start[loadstep])&(self.time<self.time_step_end[loadstep])]
        if set_zero:
                e0 = e[0]
                sigma0 = sigma[0]
        else:
            e0 = 0
            sigma0 = 0
        if 'IL' in loadstep:
            plt.semilogx(time,e-e0,**kws)
            plt.xlabel('Time (s)')
            plt.ylabel('Void ratio $e$')
        else:
            plt.semilogx(time,sigma-sigma0,**kws)
            plt.xlabel('Time (s)')
            plt.ylabel('Stress (kPa)')

if __name__ == '__main__':
    from evspy.loadsteps import LoadSteps
    load=LoadSteps()
    load.add_load_step(1e5,5,1e-8,'CRS',cv=7)
    #load.add_load_step(1e5,15,1e-6,'CRS',cv=7)
    #load.add_load_step(1e5,30,1e-8,'CRS',cv=7)
    #load.add_load_step(1e5,70,1e-6,'CRS',cv=7)
    #load.add_load_step(1e5,100,1e-8,'CRS',cv=7)
    #load.add_load_step(1e5,200,1e-8,'IL',cv=7)
    #load.add_load_step(1e8,200,1e-8,'IL',cv=7)
    #load.add_load_step(1e8,20,-1e-6,'CRS',cv=7)
    #load.add_load_step(5e6,20,-1e-8,'IL',cv=7)
    #load.add_load_step(1e7,20,0,'CRS',cv=7)
    model = distorted_isotache_model(load,sigma0=1,H=0.1,dimt=100)
    # Add a check to see if initial condition is reasonable (for instance not above reference line)
    model.initialize_decoupled(1,1,1e-10)
    model.run_iterations()