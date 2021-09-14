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

def distorted_isotaches_coupled(load,sigma0,H,erateref = 1e-5,dimt=8000,erateinits=1e-10,Cc=0.5,Cr=0.5/5,CalphaNC=0.5*0.04,k1=-8,k2=2,kf='k1',Cv=7,isotache = False,power_law=False,ref_func='semilogx',beta2=3,beta3=16,e0=2.3,dsig=0.005,de=0.0002,plotting=True,m1=0.88,b1=-0.3,b2=-5.,m2=2.,beta_nash=False,param_nash=[0.023,0.33,13],Calpha_OCRref=True,dimx=20,OpenBottom=False,sig0=1,gamman=17,gammas=27,gammaw=10,strain_definition='e/(1+e)',kdep=True,largestrain=True):
    CalphaCc=CalphaNC/Cc

    # Use Cc instead of Cc2 and skip "flexible"
    def consol_step(t,reset,Cc):
        ##### CONSOLIDATION
        
        if strain_definition=='e/(1+e0)':
            eref=e[:,0]
        elif strain_definition=='e/(1+e)':
            eref=e[:,t-1]

        if kf=='k1':
            k[:,t]=(10**k0)*np.exp(-(e[0,0]-e[:,t-1])/Ck)#k1(e[:,t-1],e[0,0],k0,Ck)
        else:
            k[:,t]=np.clip(((10**a) * e[:,t-1] ** b),1e-12,1e-4)#k2(e[:,t-1],a,b)#,kmin,kmax MUCH FASTER
        sigmaT[:,t]=sigmaT[:,t-1]+(P[t]-P[t-1])+((z[-2,t-1]-z[:,t-1])-(z[-2,t-2]-z[:,t-2]))*gammaw*(t>1)
        dHt[0]=-z[0,t-1];dHt[1:]=z[1:,t-1]-z[:-1,t-1]
        dHt[:-1]=(dHt[:-1]+dHt[1:])/2; dHt[-1]=dHt[-1]
        dHt[-2]=z[-2,t-1]-z[-3,t-1]; # 4/04/2018 This is changed to complete the drainage path leading to a correct "sum"
        if largestrain==False:
            dHt[0]=-z[0,0];dHt[1:]=z[1:,0]-z[:-1,0]
            dHt[:-1]=(dHt[:-1]+dHt[1:])/2; dHt[-1]=dHt[-1]
        #r=dt[t]/gammaw/dHt[:]**2/(rhor*(e[:,t-1]/(1+eref))/sigma[:,t-1])
        r=dt[t]/gammaw/dHt[:]**2/(Cr*(0.434)/sigma[:,t-1])
        alpha = k[2:-1,t]*r[2:-1]
        betar = -(k[3:,t]-k[1:-2,t])*r[2:-1]/4*(kdep==True)# 12/05/2018; Still a small error/approximation: dH covers 1 element but central difference for dk/dz cover 2 (and then divided by 2)
        BWcenter=np.append(np.zeros((dimx-1,1)),np.diagflat(-0.5*(alpha-betar),1),1)[:-1,:]+np.append(np.append(np.zeros((dimx-2,1)),np.diagflat(1+alpha),1),np.zeros((dimx-2,1)),1)+np.append(np.diagflat(  -0.5*(alpha+betar),-1),np.zeros((dimx-1,1)),1)[1:,:]
        BW[2:-1,1:]=BWcenter; BW[-1,-1]=1; BW[-2,-2]=1; BW[-2,-1]=0; BW[-2,-3]=0
        #Boundary conditions:
        if OpenBottom==True:
            BW[0,0]=1;BW[1,1]=1
        else:
            BW[0,0]=1+r[0]*k[0,t]; BW[0,1]=-r[0]*k[0,t]; BW[1,1]=1+r[1]*k[1,t]; BW[1,2]=-r[1]*k[1,t]#1 #r

        FWcenter=np.append(np.zeros((dimx-1,1)),np.diagflat(    0.5*(alpha-betar),1),1)[:-1,:]+np.append(np.append(np.zeros((dimx-2,1)),np.diagflat(   1-alpha),1),np.zeros((dimx-2,1)),1)+np.append(np.diagflat(  0.5*(alpha+betar),-1),np.zeros((dimx-1,1)),1)[1:,:]
        FW[2:-1,1:]=FWcenter; FW[-2,-2]=1; FW[-1,-1]=1; FW[-2,-1]=0; FW[-2,-3]=0
        if OpenBottom==True:
            FW[0,0]=1;FW[1,1]=1
        else:
            FW[0,0]=1-r[0]*k[0,t];FW[0,1]=r[0]*k[0,t]; FW[1,1]=1-r[1]*k[1,t]; FW[1,2]=r[1]*k[1,t]
        try:
            #ue[:,t]=np.linalg.solve(BW,np.dot(FW,ue[:,t-1])+boundary*(epsvprate[:,t-1]-epsswellrate[:,t-1])*dt[t]/(rhor*(e[:,t-1]/(1+eref))/sigma[:,t-1]))
            ue[:,t]=np.linalg.solve(BW,np.dot(FW,ue[:,t-1])+boundary*(-eratec[:,t-1]-erates[:,t-1])/(1+e[:,t-1])*dt[t]/(Cr*(0.434)/sigma[:,t-1]))
        except:
            print('singular')#BW)
        ue[:,t]=ue[:,t]+P[t]-P[t-1] # 1.
        if OpenBottom==True:
            ue[0,t]=0; ue[1,t]=0;
        ue[-1,t]=0;ue[-2,t]=0 #np.average([ue[-1,t],ue[-3,t]]) # 2.
        
        u[:,t]=u[:,t-1]+(ue[:,t]-ue[:,t-1])+((z[-2,t-1]-z[:,t-1])-(z[-2,t-2]-z[:,t-2]))*gammaw*(t>1)
        sigma[:,t]=sigmaT[:,t]-u[:,t]#
        sigmarate[:,t]=(sigma[:,t]-sigma[:,t-1])/dt[t]

        ### VISCOPLASTIC RESPONSE

        erate_e[1:-1,t]= Cr*0.434*(sigma[1:-1,t]-sigma[1:-1,t-1])/dt[t]/np.average([sigma[1:-1,t],sigma[1:-1,t-1]],axis=0)
        ee[:,t]=ee[:,t-1]-erate_e[:,t]*dt[t] 
        OCR_real[:,t]=sigpref[:,t-1]/sigma[:,t]
        OCR[:,t]=np.clip(np.max(np.array([sigpref[:,0],np.max(sigma[:,:t],axis=1)]),axis=0)/sigma[:,t],1.0004,np.infty)
        OCRrate[:,t] = (OCR[:,t]-OCR[:,t-1])/dt[t]
        
        erates[1:-1,t],Calphahats_real[1:-1,t-1],reset=swelling_law(erates[1:-1,t-1],OCR[1:-1,t],OCRrate[1:-1,t],Calphahats_real[1:-1,t-1],dt[t],CalphaCc*Cc2,m1,b1,m2,b2,sigmarate[1:-1,t-1],-erate_e[1:-1,t-1],reset)
        erates[0,t]=erates[1,t]
        erates[-1,t]=erates[-2,t]

        if ((sigma[:,t]<sigma[:,t-1]/1.000001)&(sigma[:,t-1]<sigma[:,t-2]/1.000001)&(np.abs(e[:,t]-e[:,t-1])>0)&(np.abs(e[:,t-2]-e[:,t-1])>0)).any():#&(np.abs(e[:,t]-e[:,t-1])>0.001):
            
            if ((load.type[load_step]=='IL')&(np.round(load.load[load_step-1],1)==np.round(load.load[load_step],1))).all():
                sigp_update[:,t]=sigp_update[:,t-1]
            else:
                for x in range(dimx):
                    if (sigma[x,t]<sigma[x,t-1]/1.000001)&(sigma[x,t-1]<sigma[x,t-2]/1.000001)&(np.abs(e[x,t]-e[x,t-1])>0)&(np.abs(e[x,t-2]-e[x,t-1])>0):#&(np.abs(e[:,t]-e[:,t-1])>0.001):
                        
                        if (load.type[load_step]=='IL')&(np.round(load.load[load_step-1],1)==np.round(load.load[load_step],1)):
                            #sigp_update[x,t]=sigpref[x,t-1]
                            sigp_update[x,t]=sigp_update[x,t-1]
                        else:
                            
                            sigp_update[x,t]=np.max(sigpref[x,:])
                            #if x ==5:
                            #    print('distortion of isotaches: '+str(np.max(sigpref[x,:]))+' '+str(sigp_update[x,t]))
                    else:
                        sigp_update[x,t]=sigp_update[x,t-1]  
        else:
            sigp_update[:,t]=sigp_update[:,t-1]
        
        e[:,t]=e[:,t-1]+(eratec[:,t-1]+erates[:,t-1])*dt[t]+(ee[:,t]-ee[:,t-1])

        eratec[1:-1,t],_,_=calc_pos_isotache(sigma[1:-1,t],e[1:-1,t],erateref = erateref,steps=20,beta2 = beta2, beta3 = beta3,Cc = Cc,Cr = Cr,CalphaNC = CalphaCc*Cc2,sigp=sigp_update[1:-1,t],e0=e0,isotache=isotache,power_law=power_law,ref_func=ref_func,beta_nash=beta_nash,param_nash=param_nash,Calpha_OCRref=Calpha_OCRref,lookup=lookup)
        
        while eratec[-2,t]>erateref:
            e[-2,t]=e[-2,t]-0.01
            eratec[1:-1,t],_,_=calc_pos_isotache(sigma[1:-1,t],e[1:-1,t],erateref = erateref,steps=20,beta2 = beta2, beta3 = beta3,Cc = Cc,Cr = Cr,CalphaNC = CalphaCc*Cc2,sigp=sigp_update[1:-1,t],e0=e0,isotache=isotache,power_law=power_law,ref_func=ref_func,beta_nash=beta_nash,param_nash=param_nash,Calpha_OCRref=Calpha_OCRref,lookup=lookup)
        eratec[:,t]=-eratec[:,t]
        erate[:,t]=(e[:,t]-e[:,t-1])/dt[t]#

        # Minimum value of OCR here increased a bit to get the erate_s response right
        
        Calphahats_real[1:-1,t]=CalphaCc*Cc2*10**b1*(np.clip(OCR[1:-1,t]-1,0.005,np.infty))**m1
        if ref_func =='semilogx':
            #sigpref[:,t] = 10**(np.log10(sigpref[:,t-1])-(e[:,t]-e[:,t-1])/(Cc-Cr)-Cr/(Cc-Cr)*np.log10(sigma[:,t]/sigma[:,t-1]))
            sigpref[:,t] = semilogintersection(e0,e[:,t],sigma[:,t],Cc,Cr)[0]
        elif ref_func =='loglog':
            Cc2=rhoc*e[1:-1,t]
            #sigpref[:,t] = 10**(np.log10(sigpref[:,t-1])-(e[:,t]-e[:,t-1])/(Cc2-Cr)-Cr/(Cc2-Cr)*np.log10(sigma[:,t]/sigma[:,t-1]))
            sigpref[:,t] = np.exp(np.log(sigpref[:,t-1])-np.log(e[:,t]/e[:,t-1])/(rhoc-rhor)-rhor/(rhoc-rhor)*np.log(sigma[:,t]/sigma[:,t-1]))
            #sigpref[:,t] = loglogintersection(rhoc,Cr,e0,sigma[:,t-1],e[:,t-1])
            if t%20==0:
                try:
                    #sigpref[:,t]=get_intersection(sigma[:,t],e[:,t],sigrangeNC,eNC,Cr)[0]
                    for x in range(1,dimx-1):
                        sigpref[x,t]=np.clip(loglogintersection(e0,e[x,t],sigma[x,t],rhoc,Cr),sigpref[x,t-1]/1.01,sigpref[x,t-1]*1.01)
                    #print(sigpref[x,t])
                    sigpref[0,t]=sigpref[1,t]
                    sigpref[-1,t]=sigpref[-2,t]
                except:
                    print('sigpref failed')
        else:
            sigpref[:,t]=get_intersection(sigma[:,t],e[:,t],sigrangeNC,eNC,Cr)[0]
            

        eps[1:,t]=(e[1:,0]-e[1:,t])/(1+e[1:,0])
        St[:-1,t]=np.cumsum(eps[:-1,t])*dH-eps[1,t]*dH/2-eps[:-1,t]*dH/2
        epscum[t]=(St[-2,t])/H
        epscumrate[t]=(St[-2,t]-St[-2,t-1])/H/dt[t]
        z[1:-1,t]=z[1:-1,0]-St[1:-1,t]
        z[0,t]=-z[2,t];z[-1,t]=z[-2,t]+z[2,t]
        globals().update(locals())
        return reset,Cc2
    if kf=='k1':
        k0,Ck = k1,k2
    else:
        a,b = k1,k2
    lookup=np.clip(2/(inversefunc(log_mitsr_strain_rate,y_values=np.arange(-100,0,0.1),domain=0.00001,args=(beta2,Cc,Cr,CalphaNC))**beta2+1),0.0,1)*CalphaNC
    time_step={}
    sigrangeNC =  10**np.arange(-2,10,0.1)
    if ref_func=='loglog':
        rhoc=Cc/e0
        rhor=Cr/e0
        eNC = np.exp(np.log(e0) - rhoc*np.log(sigrangeNC))
        Cc2 = rhoc*e0
    elif ref_func =='semilogx':
        eNC = e0 - Cc*np.log10(sigrangeNC)
        Cc2 = Cc
    elif ref_func=='flexible':
        slope_flex = Cc/m/np.log10((sig_start**-1))**(m-1)
        eNC = e0 -slope_flex*np.log10(sigrangeNC*(sig_start**-1))**m+slope_flex*np.log10((sig_start**-1))**m
        Cc2 = Cc

    #time=np.zeros((dimt)    
    eratec=np.zeros((dimx+1,dimt)); erates=np.zeros((dimx+1,dimt)); erate_e=np.zeros((dimx+1,dimt))
    e=np.zeros((dimx+1,dimt)); erate=np.zeros((dimx+1,dimt)); ee=np.zeros((dimx+1,dimt))
    OCR_real=np.zeros((dimx+1,dimt)); OCR_ref=np.zeros((dimx,dimt)); OCR=np.zeros((dimx+1,dimt))
    sigp_update=np.zeros((dimx+1,dimt))
    k=np.zeros((dimx+1,dimt)); erate_new = np.zeros((dimt)); erate_new[:]=1e-10
    u=np.zeros((dimx+1,dimt)); ue=np.zeros((dimx+1,dimt));
    dt=np.zeros((dimt)); time=np.zeros((dimt)); z=np.zeros((dimx+1,dimt));
    sigmaT=np.zeros((dimx+1,dimt)); sigma=np.zeros((dimx+1,dimt));sigmarate=np.zeros((dimx+1,dimt));
    dt=np.zeros((dimt))

    sigma[:,0]=sigma0
    e_init=(gammas-gamman)/(gamman-gammaw)
    e[:,0]=e_init
    sigpref=np.zeros((dimx+1,dimt))
    sigpref[:,0]=get_intersection(sigma[:,0],e[:,0],sigrangeNC,eNC,Cr)[0]
    #print(sigpref[:,0])
    #print(sigpref[0])
    OCR_real[:,0]=sigpref[:,0]/sigma[:,0]
    OCR[:,0]=sigpref[:,0]/sigma[:,0]
    #print(sigma[0],e[0],beta2,beta3,Cc,Cr,CalphaNC,sigpref[0],e0,isotache,power_law,ref_func)
    #print('strart')
    eratec[:,0],_,_=calc_pos_isotache(sigma[:,0],e[:,0],erateref = erateref,steps=200,beta2 = beta2, beta3 = beta3,Cc = Cc,Cr = Cr,CalphaNC = CalphaCc*Cc2,sigp=sigpref[:,0],e0=e0,isotache=isotache,power_law=power_law,ref_func=ref_func,beta_nash=beta_nash,param_nash=param_nash,dsig=dsig,Calpha_OCRref=Calpha_OCRref,lookup=lookup)#-10**get_erate(sigma[0],e[0],ei,sigi,erateii)
    #print('stop')
    erates[:,0]=erateinits
    erate[:,0]=eratec[:,0]-erates[:,0]
    Calphahatc_real=np.zeros((dimx+1,dimt))
    Calphahats_real=np.zeros((dimx+1,dimt))
    OCRrate=np.zeros((dimx+1,dimt))
    Calphahats_real[:,0]=CalphaCc*Cc2*b1*(OCR[:,0]-1)**m1
    t=0
    #start=False
    i=0
    failsafe=0
    dtmult = 1.01
    targettime=1e20
    dtfactor=1
    dtfactorinit=dtfactor
    OCR_ref=OCR_real[0]
    sigp_update[:,0]=sigpref[:,0]
    #while (t<dimt-1) and (dt>0): #(time[t]<targettime) and 
    printing=False
    load_step = 0
    next_load_step=True
    #for t in tqdm.tqdm(range(1,dimt)):
    t=0
    running=True
    ilstart=0
    reset=np.zeros((dimx-1))

    BW=np.zeros((dimx+1,dimx+1)); eps=np.zeros(([dimx+1,dimt]))
    FW=np.zeros((dimx+1,dimx+1)); St=np.zeros((dimx+1,dimt)); St[:,:]=np.nan; St[0,:]=0; St[1,:]=0; St[-1,:]=0
    r=np.zeros((dimx+1,dimt));
    epscum=np.zeros((dimt)); epscumrate=np.zeros((dimt))
    dHt=np.zeros((dimx+1))
    Test=np.zeros((dimx+1,dimt)); Test[:,:]=np.nan
    BW=np.zeros((dimx+1,dimx+1)); FW=np.zeros((dimx+1,dimx+1));
    if OpenBottom==True:
        boundary=np.zeros((dimx+1)); boundary[2:-2]=1
    else:
        boundary=np.zeros((dimx+1)); boundary[1:-2]=1
    P=np.zeros((dimt))
    dH = H/(dimx-2)
    z[1:-1,0]=np.linspace(0,H,dimx-1)
    z[0,0]=-z[2,0];z[-1,0]=z[-2,0]+z[2,0];
    Load=load.loaddf
    P[0]=np.interp(0,Load.index,Load['Load'])
    sigmaT[0,0]=sig0+H*gamman+P[0]
    sigmaT[-1,0]=sig0+P[0]
    sigmaT[1:-1,0]=sig0+((z[-2,0]-z[1:-1,0]))*gamman+P[0]
    u[1:-1,0]=(z[-2,0]-z[1:-1,0])*gamman+P[0]#sigmaT[:,0]-sig0#(z[-1,0]-z[:,0])*gammaw+P[0]-sig0#sigmaT[:,0]-0.1#(z[-1,0]-z[:,0])*gammaw+P[0]#-sig0 ==> Self-weight incl. or not...
    u[0,0]=H*gamman+P[0]
    ue[1:-1,0]=u[1:-1,0]-(z[-2,0]-z[1:-1,0])*gammaw
    sigma[:,0]=sigmaT[:,0]-u[:,0]; 

    dt[0]=1e-10;
    t=0; tincr=1.005; safety=0
    min_erates =0
    time_step={}
    time_step[load.type[load_step]+str(load_step)]=time[t]
    #print('start')
    while time[t]<targettime and t<dimt-1 and safety<dimt*2:#dimt
        safety+=1
        if np.isnan(eratec[1:-1,t]+erates[1:-1,t]).any():# or (e[1:-1,t]<0.5).any():
            #break
            t=t+1
            return time[:t],epscum[:t],epscumrate[:t],P[:t], sigma[:,:t],ue[:,:t], e[:,:t], erate[:,:t], erate_e[:,:t], eratec[:,:t], erates[:,:t], Calphahats_real[:,:t], ee[:,:t], sigpref[:,:t], time_step, OCR[:,:t], OCR_real[:,:t], sigp_update[:,:t]
            t1=t
            print(sigp_update[1:-1,t-1])
            print(eratec[1:-1,t-1])
            print(erates[1:-1,t-1])
            t=t0#np.max([t-1000,1])
            dt[t]=np.clip(dt[t]/100,0.0000001,10000)
            tincr = 1+(tincr-1)/2
            print("trigger1 at "+str(time[t1])+" reset to "+str(time[t]))
            
        elif (np.abs(erate[2:-2,t]*dt[t])>0.1).any():
            print("trigger2")
            t=np.max([t-1,1])
            dt[t]=np.min([dt[t],np.min(np.abs(0.01/erate[:,t]))])
            #print(dt[t])
            #print(erate[1:-1,t]*dt[t])


        elif (np.abs(sigmarate[2:-2,t]*dt[t]>1)).any():
            #t=np.max([t-15,1])
            tincr=1+(tincr-1)/3
            dt[t]=np.clip(dt[t]/2,0.0000001,10000)
            print("trigger3")
        elif (load.type[load_step]=='CRS')&(np.abs(P[t]-P[t-1])>0.1*P[t-1]):#(np.abs(P[t]-P[t-1])>np.clip(1.1*np.abs((P[t-1]-P[t-2])),0.01*P[t-1],10)):
            t=np.max([t-10,1])
            tincr=1+(tincr-1)/1.1
            dt[t]=np.clip(dt[t]/1.1,0.0000001,10000)
            print('trigger4')
        else:
            t=t+1
            #dt[t]=np.clip(dt[t-1]*tincr,0.0000001,np.min(np.abs(0.05/eratec[:,t-1])))
            
            ## Trial based on CRS testing
            #if (np.max(np.abs(eratec[:,t]/eratec[:,t-1]))>12)|(np.max(np.abs(erates[:,t]/erates[:,t-1]))>12): #1.2
            #    dt[t]=dt[t-1]/1.1
            #    print('Soft trigger')
            #else:
            dt[t]=dt[t-1]*tincr
            #tincr=tincr*1.0005

        time[t]=time[t-1]+dt[t]  

        if load.type[load_step]=='IL':
            P[t]=load.load[load_step]
            if ((time[t]-time_step[load.type[load_step]+str(load_step)]>=load.duration[load_step]))&(load_step<load.load_steps[-1]): 
                next_load_step=True
                reset=np.zeros((dimx-1))
        elif load.type[load_step]=='CRS':
            P[t]=P[t-1]+(erate_crs-epscumrate[t-1]*(1+e0))*P[t-1]/Cr*dt[t]
            #P[t]=P[t-1]+P[t-1]-P[t-2]
            #print(erate_crs-epscumrate[t-1]/(1+e0))

            if load.rate[load_step]>0:
                if np.average(sigma[1:-1,t-1])>=load.load[load_step]:
                    next_load_step=True
            elif load.rate[load_step]<0:
                if np.average(sigma[1:-1,t-1])<=load.load[load_step]:
                    next_load_step=True
            else:
                if (time[t]-time_step[load.type[load_step]+str(load_step)]>=load.duration[load_step]):
                    next_load_step=True
        if next_load_step:
            reset=np.zeros((dimx-1))
            t0=t
            dt[t]=1e-3
            tincr=1.01
            if len(load.type)>load_step+1:
                load_step+=1
                print('step: '+str(load_step))
                time_step[load.type[load_step]+str(load_step)]=time[t]
                if load.type[load_step]=='IL':
                    P[t]=load.load[load_step]
                elif load.type[load_step]=='CRS':
                    erate_crs = load.rate[load_step]
            next_load_step=False
        reset,Cc2=consol_step(t,reset,Cc2)
        if load.type[load_step]=='CRS':
            #print(erate_crs)
            #print(np.abs(epscumrate[t]/(1+e0)))
            #print(P[t])
            iter1=0
            maxiter=200
            if erate_crs == 0:
                reduc=1
                overshoot=True
                while (np.abs(epscumrate[t]*(1+e0))>1e-12)&(iter1<=maxiter):
                    sign = np.sign(erate_crs-epscumrate[t])
                    P[t]=P[t]+2000/reduc*(erate_crs-epscumrate[t]*(1+e0))*P[t-1]/Cr*dt[t]
                    reset,Cc2=consol_step(t,reset,Cc2)
                    #print(np.abs(epscumrate[t]/(1+e0)))
                    if np.sign(erate_crs-epscumrate[t]*(1+e0))==sign:
                        #reduc=1
                        # if you still dont get there without overshooting after 20 iterations, speed up!
                        if (iter1>20)&(overshoot):
                            reduc=reduc/1.1
                    else:
                        reduc=reduc*1.2
                        overshoot=False
                        #if iter1<5: #If you already overshoot in the first 5 iterations, better to start from scratch
                        #    reduc=reduc*10
                        #    P[t]=P[t-1]+(erate_crs-epscumrate[t-1]/(1+e0))*P[t-1]/Cr*dt[t]
                    if (iter1==20):
                        #print('######  reset')
                        reduc=5
                        P[t]=P[t-1]+(erate_crs-epscumrate[t-1]*(1+e0))*P[t-1]/Cr*dt[t]
                        reset,Cc2=consol_step(t,reset,Cc2)
                    if (not overshoot)&(iter1==100):
                        reduc=20
                        P[t]=P[t-1]+(erate_crs-epscumrate[t-1]*(1+e0))*P[t-1]/Cr*dt[t]
                        reset,Cc2=consol_step(t,reset,Cc2)
                    iter1+=1
            else:
                
                reduc=1
                overshoot=True
                while (np.abs((erate_crs/(epscumrate[t]*(1+e0))-1))>0.01)&(iter1<=maxiter):
                    sign = np.sign(erate_crs-epscumrate[t]*(1+e0))
                    P[t]=P[t]+2000/reduc*(erate_crs-epscumrate[t]*(1+e0))*P[t-1]/Cr*dt[t]
                    #print('P change '+str((erate_crs-epscumrate[t]/(1+e0))*P[t-1]/Cr*dt[t]))
                    reset,Cc2=consol_step(t,reset,Cc2)
                    if np.sign(erate_crs-epscumrate[t]*(1+e0))==sign:
                        #reduc=1
                        # if you still dont get there without overshooting after 20 iterations, speed up!
                        if (iter1>20)&(overshoot):
                            reduc=reduc/1.1
                    else:
                        reduc=reduc*1.2
                        overshoot=False
                        #if iter1<5: #If you already overshoot in the first 5 iterations, better to start from scratch
                        #    reduc=reduc*10
                        #    P[t]=P[t-1]+(erate_crs-epscumrate[t-1]/(1+e0))*P[t-1]/Cr*dt[t]
                    if (not overshoot)&(iter1==20):
                        reduc=5
                        P[t]=P[t-1]+(erate_crs-epscumrate[t-1]*(1+e0))*P[t-1]/Cr*dt[t]
                        reset,Cc2=consol_step(t,reset,Cc2)
                        #print('reset')
                    #if iter1>50:
                    #    print(erate_crs-epscumrate[t]/(1+e0))
                    #    print(reduc)
                    #    print('P change '+str(2000/reduc*(erate_crs-epscumrate[t]/(1+e0))*P[t-1]/Cr*dt[t]))
                    #print(np.abs(np.log10(erate_crs/(epscumrate[t]/(1+e0))-1)))
                    iter1+=1
            if iter1 == maxiter+1:
                print('################################# maxiter')
                print(reduc)
            #print(P[t])
        #if t%1000 == 0:
        #   print(np.abs(epscumrate[t]*(1+e0)))
        #    print(erate_crs)
        #    print(time[t])
            #print(iter1)
    return time[:t],epscum[:t],epscumrate[:t],P[:t], sigma[:,:t],ue[:,:t], e[:,:t], erate[:,:t], erate_e[:,:t], eratec[:,:t], erates[:,:t], Calphahats_real[:,:t], ee[:,:t], sigpref[:,:t], time_step, OCR[:,:t], OCR_real[:,:t], sigp_update[:,:t],z[:,:t]

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