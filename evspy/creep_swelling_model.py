import numpy as np; import pandas as pd;
from scipy.optimize import fsolve
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.special import lambertw
g=9.81; gammaw=1*g


def C_S_model_with_load(Calphahatc,Calphahats,erateci,eratesi,Cr,Cc,OCR_initial,OCR_final,dimt=100000,dtfactor=1e-7,Cv=5/3600/24/365,H=0.01,targettime=1e10,nonlinear_Calphahat=False,sigmaref=100,beta2=4,beta3=19,hypA=False):
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
    OCR_ref=np.zeros((dimt)); OCR=np.zeros((dimt))
    OCR_ref[0]=OCR_initial
    sigma=np.zeros((dimt))
    sigma[0]=sigmaref
    sigp_ref=np.zeros((dimt))
    sigp_ref[0]=sigmaref*OCR_initial
    Calphahatc_real=np.zeros((dimt))
    Calphahats_real=np.zeros((dimt))
    erateref=erateci/beta3_fit(beta3,OCR_final)
    t=0
    if nonlinear_Calphahat==True:
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
            OCR_ref[t]=sigp_ref[t-1]/sigma[t]
            OCR[t]=sigmaref*OCR_initial/sigma[t]
            if (OCR[t]<OCR_final)&(OCR[t]>1.):
                erates[t-1]=erates[t-1]*10**(5*np.log10((OCR[t]-1)/(OCR[t-1]-1))) # Viscous swelling develops and reduces at the same time while unloading
            if t==1:
                erates[t-1]=eratesi*10**(5*np.log10(np.clip((OCR[t]-1)/(OCR_final-1),0,np.infty)))
            if (OCR[t]<1.05)&hypA:
                erate_e[t]= Cc*0.434*(sigma[t]-sigma[t-1])/dt/sigma[t]
            else:
                erate_e[t]= Cr*0.434*(sigma[t]-sigma[t-1])/dt/sigma[t]
            ee[t]=ee[t-1]+erate_e[t]*dt 
            if hypA:
                e[t]=e[t-1]+(eratec[t-1]-erates[t-1])*dt*(np.abs(OCR[t]-OCR_final)<0.1)+(ee[t]-ee[t-1]) #
            else:
                e[t]=e[t-1]+(eratec[t-1]-erates[t-1])*dt+(ee[t]-ee[t-1])

            if (OCR_final < OCR_initial) and not hypA:
                eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt+eratec[t-1]*erate_e[t]*(Cc-Cr)/Cr*1/(Calphahatc_real[t-1]*0.434)*dt 
            else:
                eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt
              
            erates[t]=erates[t-1]-erates[t-1]*erates[t-1]*1/(Calphahats_real[t-1]*0.434)*dt 
            erate[t]=(e[t]-e[t-1])/dt#
            sigp_ref[t]=sigp_ref[t-1]
            sigp_ref[t]=sigp_ref[t]+sigp_ref[t]*(eratec[t]-erates[t])/(Cr*CcCrRatio-Cr)/0.434*dt

            # Calphac evolution related to eratec instead of OCRref
               
            if nonlinear_Calphahat==True:
                Calphahats_real[t]=ratio_Calphasinit(sigp_ref[t]/sigma[t])*(Calphahats/ratio_Calphasinit(OCR_final))
                Calphahatc_real[t]=Calphahatc*ratio_Calphac_f_rate(eratec[t]/erateref,beta2,beta3)/ratio_Calphac_f_rate(erateci/erateref,beta2,beta3)#
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
    return {'time':time[:t],'e':e[:t],'erate':erate[:t],'erate_c':eratec[:t],'erate_s':erates[:t],'Calphahatc':Calphahatc_real[:t],'OCR':OCR[:t],'OCR_ref':OCR_ref[:t],'erate_e':erate_e[:t]}


def Calpha_from_erate(erate,eref=1e-5,Cc=0.2,Cr=0.2/5,Calpha=0.2*0.04,beta2=3,beta3=19):
    K=(Cc-Cr)/Calpha
    R=erate/eref
    Z=-2*beta2/K*np.log(R)+1
    OCR_x=lambertw(np.exp(Z))**(1/beta2)
    return 2/(OCR_x**beta2+1)

def C_S_model_with_ocr(Calphahatc,Calphahats,erateci,eratesi,Cr,OCRtarget,Cc,dimt=100000,dtfactor=1e-7,Cv=5/3600/24/365,H=0.01,targettime=1e10,creep_coupled=False,sigmaref=100,beta2=3,beta3=19):
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
        OCR=np.zeros((dimt))
        OCR[0]=1
        sigp_ref=np.zeros((dimt))
        sigp_ref[0]=sigmaref
        Calphahatc_real=np.zeros((dimt))
        t=0
        erateref=erateci/beta3_fit(beta3,OCRtarget)
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
                OCR[t]=1/(1-np.clip(UavgAnal,0,1)+1/OCRtarget*np.clip(UavgAnal,0,1))
                if OCR[t]<OCRtarget:
                    erates[t-1]=erates[t-1]*10**(5*np.log10(np.clip(OCR[t]-1,0.0000000000001,np.infty)/(OCR[t-1]-1))) # Viscous swelling develops and reduces at the same time while unloading ==> TO BE CHECKED EXPERIMENTALLY ON VARIOUS SCALES
                if t==1:
                    erates[t-1]=eratesi*10**(5*np.log10((OCR[t]-1)/(OCRtarget-1)))
                ee[t]=-Cr*np.log10(OCRtarget)*UavgAnal
                e[t]=e[t-1]+(eratec[t-1]-erates[t-1])*dt+(ee[t]-ee[t-1])
                if creep_coupled==True:
                    eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt 
                else:
                    eratec[t]=eratec[t-1]-eratec[t-1]*eratec[t-1]*1/(Calphahatc_real[t-1]*0.434)*dt 
                erates[t]=erates[t-1]-erates[t-1]*erates[t-1]*1/(Calphahats*0.434)*dt 
                erate[t]=(e[t]-e[t-1])/dt#
                sigp_ref[t]=sigp_ref[t-1]+(OCR[t]-OCR[t-1])*sigmaref
                sigp_ref[t]=sigp_ref[t]+sigp_ref[t]*(eratec[t]-erates[t])/(Cr*CcCrRatio-Cr)/0.434*dt

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
        return {'time':time[:t],'sigma':0,'e':e[:t],'erate':erate[:t],'erate_c':eratec[:t],'erate_s':erates[:t],'Calphahatc':Calphahatc_real[:t],'OCR':OCR[:t]}

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

def C_S_relaxation(Calphahatc,Calphahats,erateci,eratesi,Cc,Cr,sigma0,dimt=100000,dtfactor=1e-7,targettime=1e10,nonlinear_Calphahat=False,swelling_isotache=True,OCR = 1,beta2=4,beta3=20.5):
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
            eratec[t]=eratec[t-1]+eratec[t-1]*erate_e[t-1]*Cc/Cr/(Calphahatc_real[t-1]*0.434)*dt #**2
            if swelling_isotache:
                erates[t]=erates[t-1]-erates[t-1]*erates[t-1]*Cc/Cr/(Calphahats_real[t-1]*0.434)*dt
            else:
                erates[t]=erates[t-1]-erates[t-1]*erates[t-1]/(Calphahats_real[t-1]*0.434)*dt

            erate_e[t]=-(eratec[t]-erates[t])
            erate[t]=erate_e[t]+eratec[t]-erates[t]
            sigma[t]=sigma[t-1]+erate_e[t]*sigma[t-1]/Cr/0.434*dt
            dt=np.clip(1/np.max([np.abs(eratec[t]),np.abs(erates[t])])*dtfactor,0,time[t]/2)
            if nonlinear_Calphahat==True:
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
    return {'time':time[:t],'e':e[:t],'erate':erate[:t],'erate_c':eratec[:t],'erate_s':erates[:t],'Calphahatc':Calphahatc_real[:t],'sigma':sigma[:t],'Calphahats':Calphahats_real[:t]}

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
def ratio_Calphacinit(OCR,beta2):
    return 2/((OCR)**beta2+1)
def ratio_Calphasinit(OCR,m=0.99,b=-1.44):
    return np.clip(np.clip(OCR-1,0.001,np.infty)**m*(10**b),1e-100,1e100)

def C_S_CRS(Cc,Cr,sigmap,eratec0,CalphaNC,erateref=2e-4,Cc_reduction=2.3,Cc_f_OCR=False,erates0=-1,erate = 1,sigma0=265,sigmaf=160,b2=-5.5,m2=2.5,beta2=3,beta3=18,estart=1,m1=0.88,b1=-1.4,dtfactor=1):
    Cc0 = Cc
    dimt=100000
    time=np.zeros((dimt))
    sigp_ref=np.zeros((dimt))
    eratec=np.zeros((dimt)); erates=np.zeros((dimt)); erate_e=np.zeros((dimt))
    e=np.zeros((dimt)); sigma=np.zeros((dimt))
    OCRrate=np.zeros((dimt))
    e[0]=estart
    OCR=np.zeros((dimt))
    OCRref=np.zeros((dimt))
    sigp_ref[0]=sigmap
    OCR[:]=np.clip(sigmap/sigma0,1.01,np.infty)
    OCRref[:]=np.clip(sigp_ref/sigma0,1.01,np.infty)
    dt=np.abs(1/erate/1000000)*dtfactor
    sigma[0]=sigma0
    eratec[0]=eratec0
    if erates0==-1:
        erates[0]=10**(b2)*((OCR[0]-1)**m2)
    else:
        erates[0]=erates0
    Calphahatc_real=np.zeros((dimt))
    Calphahats_real=np.zeros((dimt))
    Calphahatc_real[0]=ratio_Calphacinit(sigp_ref[0]/sigma[0],beta2)*CalphaNC
    Calphahats_real[0]=ratio_Calphasinit(sigp_ref[0]/sigma[0],m=m1,b=b1)*CalphaNC
    t=0
    stop=True
    while stop & (t<dimt-2):
        if t>1:
            if (np.abs(eratec[t]/eratec[t-1])>1.1)|(np.abs(erates[t]/erates[t-1])>1.1):
                dt=dt/1.1
            else:
                dt=dt*1.01
        t+=1
        time[t]=time[t-1]+dt
        OCR[t]=np.clip(sigmap/sigma[t-1],1.01,np.infty)
        OCRref[t]=np.clip(sigp_ref[t-1]/sigma[t-1],1.01,np.infty)
        OCRrate[t] = (OCR[t]-OCR[t-1])/dt
        eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt+eratec[t-1]*erate_e[t-1]*(Cc-Cr)/Cr*1/(Calphahatc_real[t-1]*0.434)*dt #**2
        erates[t]=erates[t-1]+m2*OCRrate[t]/(OCR[t]-1)*erates[t-1]*dt-erates[t-1]*erates[t-1]/(Calphahats_real[t-1]*0.434)*dt
        erate_e[t] = -erate-eratec[t]+erates[t]
        sigma[t]=sigma[t-1]+(erate_e[t]/Cr)*sigma[t-1]/0.434*dt
        e[t]=e[t-1]+erate*dt
        sigmap = np.max([sigma[t],sigmap])
        sigp_ref[t]=sigp_ref[t-1]+sigp_ref[t-1]*(eratec[t]-erates[t])/(Cc-Cr)/0.434*dt
        
        Calphahatc_real[t]=CalphaNC*np.clip(ratio_Calphac_f_rate(eratec[t]/erateref,beta2,beta3),0.1,100)
        #if sigmaf>sigma0:
        #    Calphahatc_real[t]=CalphaNC*np.clip(ratio_Calphac_f_rate(eratec[t]/erateref,beta2,beta3),0.1,100)
        #else:
        #    Calphahatc_real[t]=np.clip(FDM.ratio_Calphacinit(sigp_ref[t]/sigma[t],beta2),0.1,1)*CalphaNC   
        Calphahats_real[t]=np.clip(ratio_Calphasinit(OCR[t],m=m1,b=b1),0.05,100)*CalphaNC
        
        if sigma0>sigmaf:
            stop = (sigma[t]>sigmaf)
        else:
            stop = sigma[t]<sigmaf
        if Cc_f_OCR:
            Cc=Cc0-(Cc0-Cc0/Cc_reduction)/(1+np.exp(-10*(OCR[t]-1.2)))
    return {'time':time[:t],'sigma':sigma[:t],'e':e[:t],'erate_c':eratec[:t],'erate_e':erate_e[:t],'erate_s':erates[:t],'Calphahatc':Calphahatc_real[:t],'Calphahats':Calphahats_real[:t]}


    