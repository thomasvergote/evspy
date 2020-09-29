import numpy as np; import pandas as pd;
from scipy.optimize import fsolve
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.special import lambertw
g=9.81; gammaw=1*g


def C_S_model_with_load(Calphahatc,Calphahats,erateci,eratesi,Cr,Cc,OCR_initial,OCR_final,dimt=100000,dtfactor=1e-7,Cv=5/3600/24/365,H=0.01,targettime=1e10,creep_coupled=False,sigmaref=100,beta2=4,beta3=19,hypA=False):
        '''
    Parameters
    ----------
    Calphahatc : float
        secondary compression index (creep) after unloading at OCR_final
    
    Returns
    -------
    
    '''
    CcCrRatio = Cc/Cr
    time=np.zeros((dimt))
    eratec=np.zeros((dimt)); erates=np.zeros((dimt)); erate_e=np.zeros((dimt))
    e=np.zeros((dimt)); erate=np.zeros((dimt)); ee=np.zeros((dimt))
    eratec[0]=erateci
    erates[0]=0#eratesi
    erate[0]=eratec[0]-erates[0]
    dt=1e-20
    OCR_real=np.zeros((dimt)); OCR_ref=np.zeros((dimt))
    OCR_real[0]=OCR_initial
    sigma=np.zeros((dimt))
    sigma[0]=sigmaref
    sigp_Calpha=np.zeros((dimt))
    sigp_Calpha[0]=sigmaref*OCR_initial
    Calphahatc_real=np.zeros((dimt))
    Calphahats_real=np.zeros((dimt))
    erateref=erateci/beta3_fit(beta3,OCR_final)
    t=0
    if creep_coupled==True:
        Calphahatc_real[0]=Calphahatc#*ratio_Calphacinit(OCR_initial,beta2)/ratio_Calphacinit(OCR_final,beta2)
        Calphahats_real[0]=Calphahatc*ratio_Calphasinit(OCR_initial)/ratio_Calphasinit(OCR_final)
    else:
        Calphahatc_real[0]=Calphahatc
        Calphahats_real[0]=Calphahats
    #start=False
    i=0
    failsafe=0
    dtmult = 1.01
    dtfactorinit=dtfactor
    while (time[t]<targettime) and (t<dimt-1) and (dt>0):
        if  (dt>0)&(eratec[t]>-0.01)&(erates[t]>-0.01)&(np.abs(eratec[t]*dt)<1e-2)&(np.abs(erates[t]*dt)<1e-2):
            dtfactor=dtfactor * dtmult
            t+=1
            #print(dt)
            time[t]=time[t-1]+dt
            time2,UavgAnal=Consol_Terzaghi_Uavg_vertical(Cv,H,t=time[t],dimt=-1)
            sigma[t]=sigmaref-(sigmaref-(OCR_initial/OCR_final*sigmaref))*np.clip(UavgAnal,0,1)
            OCR_real[t]=sigp_Calpha[t-1]/sigma[t]
            OCR_ref[t]=sigmaref*OCR_initial/sigma[t]
            if (OCR_ref[t]<OCR_final)&(OCR_ref[t]>1.01):
                erates[t-1]=erates[t-1]*10**(5*np.log10((OCR_ref[t]-1)/(OCR_ref[t-1]-1))) # Viscous swelling develops and reduces at the same time while unloading
                
            if t==1:
                erates[t-1]=eratesi*10**(5*np.log10(np.clip((OCR_ref[t]-1)/(OCR_final-1),0,np.infty)))
            if OCR_real[t]<1.05:
                erate_e[t]= Cr*0.434*(sigma[t]-sigma[t-1])/dt/sigma[t]
                if hypA:
                    erate_e[t]= Cc*0.434*(sigma[t]-sigma[t-1])/dt/sigma[t]
            else:
                erate_e[t]= Cr*0.434*(sigma[t]-sigma[t-1])/dt/sigma[t]
            ee[t]=ee[t-1]+erate_e[t]*dt 
            e[t]=e[t-1]+(eratec[t-1]-erates[t-1])*dt+(ee[t]-ee[t-1])
            if creep_coupled==True:
                eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt 
                if OCR_final < OCR_initial: 
                    if hypA:
                        eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt
                    else:
                        eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt+eratec[t-1]*erate_e[t]*(Cc-Cr)/Cr*1/(Calphahatc_real[t-1]*0.434)*dt 

            else:
                eratec[t]=eratec[t-1]-eratec[t-1]*eratec[t-1]*1/(Calphahatc_real[t-1]*0.434)*dt 
            erates[t]=erates[t-1]-erates[t-1]*erates[t-1]*1/(Calphahats_real[t-1]*0.434)*dt 
            erate[t]=(e[t]-e[t-1])/dt#
            sigp_Calpha[t]=sigp_Calpha[t-1]
            sigp_Calpha[t]=sigp_Calpha[t]+sigp_Calpha[t]*(eratec[t]-erates[t])/(Cr*CcCrRatio-Cr)/0.434*dt
            if creep_coupled==True:
                Calphahatc_real[t]=Calphahatc*ratio_Calphac_f_rate(eratec[t]/erateref,beta2,beta3)/ratio_Calphac_f_rate(erateci/erateref,beta2,beta3)#
                Calphahats_real[t]=ratio_Calphasinit(sigp_Calpha[t]/sigma[t])*(Calphahats/ratio_Calphasinit(OCR_final))
            else:
                Calphahatc_real[t]=Calphahatc
                Calphahats_real[t]=Calphahats
            dt=np.clip(1/np.abs(eratec[t]-erates[t])*dtfactor,1e-100,time[t]/2)
        else:
            t = 2
            failsafe += failsafe
            dtfactor = dtfactorinit / 10
            dtmult = 1+(dtmult-1) / 2
            dtfactorinit = dtfactorinit/10
            dt=1e-100
    if failsafe > 9:
        print(Calphahatc,Calphahats,erateci,eratesi,Cr)
    return time[:t],e[:t],erate[:t],eratec[:t],erates[:t],Calphahatc_real[:t],OCR_ref[:t],erate_e[:t]#,OCR_ref[:t]


def Calpha_from_erate(erate,eref=1e-5,Cc=0.2,Cr=0.2/5,Calpha=0.2*0.04,beta2=3,beta3=19):
    K=(Cc-Cr)/Calpha
    R=erate/eref
    Z=-2*beta2/K*np.log(R)+1
    OCR_x=lambertw(np.exp(Z))**(1/beta2)
    return 2/(OCR_x**beta2+1)

def C_S_model_with_ocr(Calphahatc,Calphahats,erateci,eratesi,Cr,OCR,Cc,dimt=100000,dtfactor=1e-7,Cv=5/3600/24/365,H=0.01,targettime=1e10,creep_coupled=False,sigmaref=100,beta2=3,beta3=19):
    with np.errstate(divide='ignore', invalid='ignore'):
        CcCrRatio=Cc/Cr
        dtfactorinit=dtfactor
        time=np.zeros((dimt))
        eratec=np.zeros((dimt)); erates=np.zeros((dimt))
        e=np.zeros((dimt)); erate=np.zeros((dimt)); ee=np.zeros((dimt))
        eratec[0]=erateci
        erates[0]=0
        erate[0]=eratec[0]-erates[0]
        dt=1e-20
        OCR_real=np.zeros((dimt))
        OCR_real[0]=1
        sigp_Calpha=np.zeros((dimt))
        sigp_Calpha[0]=sigmaref
        Calphahatc_real=np.zeros((dimt))
        t=0
        erateref=erateci/beta3_fit(beta3,OCR)
        if creep_coupled==True:
            Calphahatc_real[0]=Calphahatc
        else:
            Calphahatc_real[0]=Calphahatc
        #start=False
        i=0
        failsafe=0
        dtmult = 1.01
        while (time[t]<targettime*1.5) and (t<dimt-1) and (failsafe<10):
            if  (dt>0)&(eratec[t]>-0.01)&(erates[t]>-0.01)&(np.abs(eratec[t]*dt)<1e-2)&(np.abs(erates[t]*dt)<1e-2):
                dtfactor=dtfactor * dtmult
                t+=1
                #print(dt)
                time[t]=time[t-1]+dt
                time2,UavgAnal=Consol_Terzaghi_Uavg_vertical(Cv,H,t=time[t])
                OCR_real[t]=1/(1-np.clip(UavgAnal,0,1)+1/OCR*np.clip(UavgAnal,0,1))
                if OCR_real[t]<OCR:
                    erates[t-1]=erates[t-1]*10**(5*np.log10(np.clip(OCR_real[t]-1,0.0000000000001,np.infty)/(OCR_real[t-1]-1))) # Viscous swelling develops and reduces at the same time while unloading ==> TO BE CHECKED EXPERIMENTALLY ON VARIOUS SCALES
                if t==1:
                    erates[t-1]=eratesi*10**(5*np.log10((OCR_real[t]-1)/(OCR-1)))
                ee[t]=-Cr*np.log10(OCR)*UavgAnal
                e[t]=e[t-1]+(eratec[t-1]-erates[t-1])*dt+(ee[t]-ee[t-1])
                if creep_coupled==True:
                    eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt 
                else:
                    eratec[t]=eratec[t-1]-eratec[t-1]*eratec[t-1]*1/(Calphahatc_real[t-1]*0.434)*dt 
                erates[t]=erates[t-1]-erates[t-1]*erates[t-1]*1/(Calphahats*0.434)*dt 
                erate[t]=(e[t]-e[t-1])/dt#
                sigp_Calpha[t]=sigp_Calpha[t-1]+(OCR_real[t]-OCR_real[t-1])*sigmaref
                sigp_Calpha[t]=sigp_Calpha[t]+sigp_Calpha[t]*(eratec[t]-erates[t])/(Cr*CcCrRatio-Cr)/0.434*dt

                if creep_coupled==True:
                    Calphahatc_real[t]=Calphahatc*ratio_Calphac_f_rate(eratec[t]/erateref,beta2,beta3)/ratio_Calphac_f_rate(erateci/erateref,beta2,beta3)#
                else:
                    Calphahatc_real[t]=Calphahatc
                dt=np.clip(1/np.abs(eratec[t]-erates[t])*dtfactor,1e-100,time[t]/2)
            else:
                t = 2
                failsafe += failsafe
                dtfactor = dtfactorinit / 10
                dtmult = 1+(dtmult-1) / 2
                dtfactorinit = dtfactorinit/10
                dt=1e-100
        if failsafe > 9:
            print(Calphahatc,Calphahats,erateci,eratesi,Cr)
        return time[:t],e[:t],erate[:t],eratec[:t],erates[:t],Calphahatc_real[:t],OCR_real[:t]

def ratio_Calphacinit(OCR,beta2):
    return 2/((OCR)**beta2+1)
def ratio_Calphasinit(OCR,m=0.99,b=-1.44):
    return np.clip(np.clip(OCR-1,0.001,np.infty)**m*(10**b),1e-100,1e100)
def ratio_Calphac_f_rate(eratec_ratio,beta2,beta3):
    return 2/((eratec_ratio)**(-beta2/beta3)+1)
def Consol_Terzaghi_Uavg_vertical(Cv,H,t=-1,dimt=1000,dt=1,time=-1):
    '''
    Parameters
    ----------
    Cv : float
        coefficient of vertical consolidation in m/s**2
    '''
    if dimt==-1:
        time=time
    else:
        time=np.arange(0,dimt,dt)
    if t>0:
        Tv=t*Cv/((H)**2)
    else:
        Tv=time*Cv/((H)**2)
    Tv=np.clip(Tv,0,10)
    UavgAnalV=np.sqrt(4*Tv/np.pi)/((1+(4*Tv/np.pi)**2.8)**0.179)
    return time,UavgAnalV

def C_S_relaxation(Calphahatc,Calphahats,erateci,eratesi,Cc,Cr,sigma0,dimt=100000,dtfactor=1e-7,targettime=1e10,creep_coupled=True,Calpha_coupled=False,swelling_isotache=True,OCR = 1,beta2=4,beta3=20.5):
    time=np.zeros((dimt))
    eratec=np.zeros((dimt)); erates=np.zeros((dimt)); erate_e=np.zeros((dimt))
    e=np.zeros((dimt)); erate=np.zeros((dimt))
    sigma=np.zeros((dimt)); Calphahatc_real=np.zeros((dimt)); Calphahats_real=np.zeros((dimt))
    Calphahatc_real[0]=Calphahatc
    Calphahats_real[0]=Calphahats
    sigp=np.zeros((dimt))
    sigp[0]=sigma0*OCR
    eratec[0]=erateci
    erates[0]=eratesi
    erate[0]=0
    dt=1e-20
    sigma[0]=sigma0
    erateref=erateci/beta3_fit(beta3,OCR)
    t=0
    i=0
    failsafe=0
    dtmult = 1.01
    dtfactorinit=dtfactor
    while (time[t]<targettime) and (t<dimt-1) and (dt>0):
        if  (dt>0)&(eratec[t]>-0.01)&(erates[t]>-0.01)&(np.abs(eratec[t]*dt)<1e-2)&(np.abs(erates[t]*dt)<1e-2):
            dtfactor=dtfactor * dtmult
            t+=1
            time[t]=time[t-1]+dt   
            if creep_coupled:
                eratec[t]=eratec[t-1]+eratec[t-1]*erate_e[t-1]*Cc/Cr/(Calphahatc_real[t-1]*0.434)*dt #**2
            else:
                eratec[t]=eratec[t-1]-eratec[t-1]*eratec[t-1]*Cc/Cr/(Calphahatc_real[t-1]*0.434)*dt
            if swelling_isotache:
                erates[t]=erates[t-1]-erates[t-1]*erates[t-1]*Cc/Cr/(Calphahats_real[t-1]*0.434)*dt
            else:
                erates[t]=erates[t-1]-erates[t-1]*erates[t-1]/(Calphahats_real[t-1]*0.434)*dt

            erate_e[t]=-(eratec[t]-erates[t])
            erate[t]=erate_e[t]+eratec[t]-erates[t]
            sigma[t]=sigma[t-1]+erate_e[t]*sigma[t-1]/Cr/0.434*dt
            dt=np.clip(1/np.max([np.abs(eratec[t]),np.abs(erates[t])])*dtfactor,0,time[t]/2)
            if Calpha_coupled==True:
                sigp[t]=sigp[0]/(10**((Cr/(Cc-Cr))*np.log10(sigma[t]/sigma0)))
                Calphahatc_real[t]=Calphahatc*ratio_Calphac_f_rate(eratec[t]/erateref,beta2,beta3)/ratio_Calphac_f_rate(erateci/erateref,beta2,beta3)#
                
                Calphahats_real[t]=ratio_Calphasinit(sigp[t]/sigma[t])*(Calphahats/ratio_Calphasinit(OCR))
            else:
                Calphahatc_real[t]=Calphahatc
                Calphahats_real[t]=Calphahats
        else:
            t = 2
            failsafe += failsafe
            dtfactor = dtfactorinit / 10
            dtmult = 1+(dtmult-1) / 2
            dtfactorinit = dtfactorinit/10
            dt=1e-100
    if failsafe > 9:
        print(Calphahatc,Calphahats,erateci,eratesi,Cr)
    return time[:t],e[:t],erate[:t],eratec[:t],erates[:t],sigma[:t],Calphahatc_real[:t],Calphahats_real[:t]
def mitsr_strain_rate(OCR,beta2,Cc,Cr,Calpha):
    return (np.sqrt(1/OCR)*np.exp((1-(OCR)**beta2)/(2*beta2)))**((Cc-Cr)/Calpha)

def beta2_fit(beta2,OCR):
    return 2/((OCR)**beta2+1)
def beta2_min_fit(beta2,minimum,OCR):
    return np.clip(2/((OCR)**beta2+1),minimum,1)
def beta3_fit(beta3,OCR):
    return (1/(OCR))**beta3
def power_law(OCR,m,b):
    return np.clip((OCR-1)**m*(10**b),1e-100,1e100)
def beta_nash(OCR,betamin,b0,a):
    b=np.clip(1/OCR,b0,1)
    mu = (np.pi/2)*((b-b0)/(1-b0))
    return np.nan_to_num(betamin+(1-betamin)*(np.sin(mu))**a)