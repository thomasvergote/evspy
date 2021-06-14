import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pynverse import inversefunc
import lmfit
from scipy.special import lambertw


from evspy.helper_functions import line_intersection, get_intersection # Refactoring needed for get_intersection?
from evspy.creep_swelling_model import beta3_fit, beta2_fit
# Strain rate relations

def Calpha_from_erate(erate,eref=1e-5,Cc=0.2,Cr=0.2/5,Calpha=0.2*0.04,beta2=3,beta3=19):
    K=(Cc-Cr)/Calpha
    try:
        erate.shape[1]
        erate=erate[0]
    except:
        pass
    R=erate/eref
    Z=-2*beta2/K*np.log(R)+1
    OCR_x=lambertw(np.exp(Z))**(1/beta2)
    return 2/(np.array([np.float(i) for i in OCR_x])**beta2+1)

def mitsr_strain_rate(OCR,beta2,Cc,Cr,Calpha):
    return (np.sqrt(1/OCR)*np.exp((1-(OCR)**beta2)/(2*beta2)))**((Cc-Cr)/Calpha)

def log_mitsr_strain_rate(OCR,beta2,Cc,Cr,Calpha):
    return np.log10((np.sqrt(1/OCR)*np.exp((1-(OCR)**beta2)/(2*beta2)))**((Cc-Cr)/Calpha))

def get_sigref(sigma,e,Cc,Cr,e0,loglog=False):
    if loglog:
        rhor=Cr/e0
        rhoc=Cc/e0
        X,Y=line_intersection(((np.log(sigma),np.log(e)),(5,np.log(e)-rhor*np.log(np.exp(5)/sigma))),((0,np.log(e0)),(5,np.log(e0)-rhoc*5)))
        return np.exp(X),np.exp(Y)
    else:
        X,Y=line_intersection(((np.log10(sigma),e),(5,e-Cr*np.log10(10**5/sigma))),((0,e0),(5,e0-Cc*5)))
        return 10**X,Y
def create_isotache(erateref = 1e-5,beta2 = 4,beta3 = 25,Cc = 0.2,Cr = 0.2 /5,CalphaNC = 0.2 * 0.04,sigp=1.,e0=1.5,isotache=False,power_law=False,ref_func='semilogx',m=1,beta_nash=False,m0=-1.22,b0=-1.14,param_nash=[0.02952926,0.02696060,9.03771812],dsig=0.05,Calpha_OCRref=True,beta2beta3=False):
    #lookup=np.clip(2/(inversefunc(log_mitsr_strain_rate,y_values=np.arange(-100,0,0.1),domain=0.00001,args=(beta2,Cc,Cr,CalphaNC))**beta2+1),0.0,1)*CalphaNC
    if isotache:
        sigp=1
    sigrange = 10**np.arange(-1,3,dsig)
    sigrangeNC =  10**np.arange(-3,10,dsig)
    sig_start = sigrangeNC[0]
    OCR = np.clip((sigp)/(sigrange),1,np.infty)
    if ref_func=='loglog':
        rhoc=Cc/e0
        rhor=Cr/e0
        eNC = np.exp(np.log(e0) - rhoc*np.log(sigrangeNC))
        eUC0 = np.exp(np.log(e0) +(sigrange<sigp)*(- rhoc*np.log(sigp) - (-rhor*np.log(sigp)+rhor*np.log(sigrange))) - (sigrange>=sigp)*rhoc*np.log(sigrange))
    elif ref_func =='semilogx':
        eNC = e0 - Cc*np.log10(sigrangeNC)
        eUC0 = e0 +(sigrange<sigp)*(- Cc*np.log10(sigp) - (-Cr*np.log10(sigp)+Cr*np.log10(sigrange))) - (sigrange>=sigp)*Cc*np.log10(sigrange)
    elif ref_func=='flexible':
        slope_flex = Cc/m/np.log10((sig_start**-1))**(m-1)
        eNC = e0 -slope_flex*np.log10(sigrangeNC*(sig_start**-1))**m+slope_flex*np.log10((sig_start**-1))**m
        eUC0 = e0 +(sigrange<sigp)*((-slope_flex*np.log10(sigp*(sig_start**-1))**m+slope_flex*np.log10((sig_start**-1))**m) - 
                                    (-Cr*np.log10(sigp)+Cr*np.log10(sigrange))) - (sigrange>=sigp)*(slope_flex*np.log10(sigrange*(sig_start**-1))**m-slope_flex*np.log10((sig_start**-1))**m)
    if isotache:
        eratei = mitsr_strain_rate(OCR,beta2,Cc,Cr,CalphaNC)*erateref
    else:
        eratei = beta3_fit(beta3,OCR)*erateref
    if power_law:
        Calphac = np.clip(power_law(np.clip(OCR,1.01,np.infty),m0,b0),0,1)*CalphaNC
    elif beta_nash:
        Calphac = np.clip(beta_nash(OCR,param_nash[0],param_nash[1],param_nash[2]),0,1)*CalphaNC
    else:
        if isotache:
            Calphac = np.clip(beta2_fit(beta2,OCR),0,1)*CalphaNC
        else:
            if beta2beta3:
                Calphac = np.clip(2/((eratei/erateref)**(-beta2/beta3)+1),0.1,1)*CalphaNC
            else:
                #Calphac = np.clip(beta2_fit(beta2,OCR),0,1)*CalphaNC
                Calphac = CalphaNC*Calpha_from_erate(eratei,eref=erateref,Cc=Cc,Cr=Cr,Calpha=CalphaNC,beta2=beta2,beta3=beta3)
            
    df=pd.DataFrame()
    df['sigma']=sigrange
    df['e']=eUC0
    eUC=eUC0
    df['erate']=eratei
    sigp0 = np.clip(sigp,sigrange,np.infty)
    OCR = np.clip((sigp)/(sigrange),1,np.infty)
    sigp = sigp0
    for i in range(500):
        deltae = Calphac/6 #0.25 log cycle down
        eUC = eUC-deltae
        eratei=10**(np.log10(eratei)-deltae/Calphac)
        #eratei = np.clip(10**(np.log10(eratei)-deltae/Calphac),1e-1000,1)
        if Calpha_OCRref:
            sigp = get_intersection(sigrange,eUC,sigrangeNC,eNC,Cr)[0]#(sigp+deltae/(Cc-Cr))
        OCR = sigp / sigrange
        if power_law:
            Calphac = np.clip(power_law(np.clip(OCR,1.01,np.infty),m0,b0),0,1)*CalphaNC
        elif beta_nash:
            Calphac = np.clip(beta_nash(OCR,param_nash[0],param_nash[1],param_nash[2]),0,1)*CalphaNC #param_nash=[betamin,b0,a]
        else:
            if isotache:
                Calphac = np.clip(beta2_fit(beta2,OCR),0,1)*CalphaNC
            else:
                if beta2beta3:
                    Calphac = np.clip(2/((eratei/erateref)**(-beta2/beta3)+1),0.1,1)*CalphaNC
                else:
                    #Calphac = np.clip(beta2_fit(beta2,OCR),0,1)*CalphaNC
                    Calphac = CalphaNC*Calpha_from_erate(eratei,eref=erateref,Cc=Cc,Cr=Cr,Calpha=CalphaNC,beta2=beta2,beta3=beta3)
                
        dftemp=pd.DataFrame()
        dftemp['sigma']=sigrange
        dftemp['e']=eUC
        dftemp['erate']=eratei
        df=df.append(dftemp)
    eUC = eUC0
    sigp = sigp0
    OCR = sigp / sigrange
    if isotache:
        eratei = mitsr_strain_rate(OCR,beta2,Cc,Cr,CalphaNC)*erateref
    else:
        eratei = beta3_fit(beta3,OCR)*erateref
    if power_law:
        Calphac = np.clip(power_law(np.clip(OCR,1.01,np.infty),m0,b0),0,1)*CalphaNC
    elif beta_nash:
        Calphac = np.clip(beta_nash(OCR,param_nash[0],param_nash[1],param_nash[2]),0,1)*CalphaNC #param_nash=[betamin,b0,a]
    else:
        if isotache:
            Calphac = np.clip(beta2_fit(beta2,OCR),0,1)*CalphaNC
        else:
            if beta2beta3:
                Calphac = np.clip(2/((eratei/erateref)**(-beta2/beta3)+1),0.1,1)*CalphaNC
            else:
                #Calphac = np.clip(beta2_fit(beta2,OCR),0,1)*CalphaNC
                Calphac = CalphaNC*Calpha_from_erate(eratei,eref=erateref,Cc=Cc,Cr=Cr,Calpha=CalphaNC,beta2=beta2,beta3=beta3)
    for i in range(500):
        deltae = -Calphac/6 #0.25 log cycle down
        eUC = eUC-deltae
        eratei=10**(np.log10(eratei)-deltae/Calphac)
        #eratei = np.clip(10**(np.log10(eratei)-deltae/Calphac),1e-1000,1)
        if Calpha_OCRref:
            #sigp = get_intersection(sigrange,eUC,sigrangeNC,eNC,Cr)[0]#
            sigp = (sigp+deltae/(Cc-Cr))
        OCR = sigp / sigrange
        if power_law:
            Calphac = np.clip(power_law(np.clip(OCR,1.01,np.infty),m0,b0),0,1)*CalphaNC
        elif beta_nash:
            Calphac = np.clip(beta_nash(OCR,param_nash[0],param_nash[1],param_nash[2]),0,1)*CalphaNC #param_nash=[betamin,b0,a]
        else:
            if isotache:
                Calphac = np.clip(beta2_fit(beta2,OCR),0,1)*CalphaNC
            else:
                if beta2beta3:
                    Calphac = np.clip(2/((eratei/erateref)**(-beta2/beta3)+1),0.1,1)*CalphaNC
                else:
                    #Calphac = np.clip(beta2_fit(beta2,OCR),0,1)*CalphaNC
                    Calphac = CalphaNC*Calpha_from_erate(eratei,eref=erateref,Cc=Cc,Cr=Cr,Calpha=CalphaNC,beta2=beta2,beta3=beta3)
                

        dftemp=pd.DataFrame()
        dftemp['sigma']=sigrange
        dftemp['e']=eUC
        dftemp['erate']=eratei
        df=df.append(dftemp) 
    return df,eUC,eUC0,eNC,sigrange,sigrangeNC

