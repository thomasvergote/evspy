import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pynverse import inversefunc
#import lmfit
from scipy.special import lambertw


from evspy.helper_functions import line_intersection, get_intersection # Refactoring needed for get_intersection?
from evspy.single_stage_model.creep_swelling_model import beta3_fit, beta2_fit, power_law, beta_nash
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

def calc_pos_isotache(sigma,e,erateref = 1e-5,beta2 = 4,beta3 = 25,Cc = 0.2,Cr = 0.2 /5,CalphaNC = 0.2 * 0.04,sigp=100.,e0=1.5,isotache=False,power_law=False,ref_func='semilogx', Calpha_OCRref=True,m=1,m0=-1.22,b0=-1.14,dsig=0.05,steps=300,beta_nash=False,param_nash=[0.02952926,0.02696060,9.03771812],lookup=[],beta2beta3=False):
    if len(lookup)==0:
        lookup=np.clip(2/(inversefunc(log_mitsr_strain_rate,y_values=np.arange(-100,0,0.1),domain=0.00001,args=(beta2,Cc,Cr,CalphaNC))**beta2+1),0.0,1)*CalphaNC
    
        
def make_contour(df,eUC,eUC0,eNC,sigrange,num=20,figsize=(8,5),vmin=-65,vmax=-5,colorbar=False):
    sigi=sigrange
    ei=10**np.arange(np.log10(0.5),np.log10(3),0.0005)
    erateii=griddata((df['sigma'],df['e']),np.log10(df['erate']),xi=(sigi[None,:],ei[:,None]))
    for i in range(len(ei)):
        for j in range(len(sigi)):
            if ei[i]>eUC[sigrange==sigrange[j]][0]:
                erateii[i,j]=np.nan
    
    plt.figure(figsize=figsize)
    ax=plt.subplot(111)
    #ax.annotate(r'$C_c$',xy=(10,1.75),xytext=(50,1.55))
    #ax.annotate(r'reference',xy=(10,1.75),xytext=(15,1.73))
    #ax.annotate(r'isotache',xy=(10,1.75),xytext=(22,1.69))
    #ax.annotate(r'$C_r$',xy=(4,1.6),xytext=(4,1.7))
    cs=plt.contour(sigi,ei,np.clip(erateii,vmin,vmax),num,vmax=vmax,vmin=vmin)
    plt.semilogx(sigrange,eNC,'k--')
    plt.plot(sigrange,eUC0,'k')
    if colorbar:
        plt.colorbar(label=r'$log_{10}(\dot{e}_c)$')
    plt.ylim(1.4,1.8)
    plt.xlim(2,100)
    plt.tight_layout()
    plt.xlabel("Stress, $\sigma_v'$ (kPa)"); plt.ylabel('Void ratio, $e$')#
    return cs    
    

from functools import wraps
import inspect

def instanceVariables(func):
    def returnFunc(*args, **kwargs):
        selfVar = args[0]

        argSpec = inspect.getargspec(func)
        argumentNames = argSpec[0][1:]
        defaults = argSpec[3]
        if defaults is not None:
            defaultArgDict = dict(zip(reversed(argumentNames), reversed(defaults)))
            selfVar.__dict__.update(defaultArgDict)

        argDict = dict(zip(argumentNames, args[1:]))
        selfVar.__dict__.update(argDict)


        validKeywords = set(kwargs) & set(argumentNames)
        kwargDict = {k: kwargs[k] for k in validKeywords}
        selfVar.__dict__.update(kwargDict)

        func(*args, **kwargs)

    return returnFunc

class base_isotache_model:

    @instanceVariables
    def __init__(self,load,sigma0,H,dimt=8000,erateref = 1e-5,erateinits=1e-10,
                    beta2 = 4,beta3 = 25,Cc = 0.2, Cr = 0.2 /5,CalphaNC = 0.2 * 0.04,
                    Cv=7,sigp=100.,e0=1.5,isotache=False,use_power_law=False,ref_func='semilogx', 
                    e_init=1.9,Calpha_OCRref=True,m0=-1.22,b0=-1.14,
                    m1=0.88,b1=-0.3,b2=-5.,m2=2.,dsig=0.05,steps=300,
                    use_beta_nash=False,param_nash=[0.02952926,0.02696060,9.03771812],
                    beta2beta3=False):
        self.eratec=np.zeros((dimt)); self.erates=np.zeros((dimt)); self.erate_e=np.zeros((dimt))
        self.e=np.zeros((dimt)); self.erate=np.zeros((dimt)); self.ee=np.zeros((dimt))
        self.OCR_real=np.zeros((dimt)); self.OCR_ref=np.zeros((dimt)); self.OCR=np.zeros((dimt))
        self.sigp_update=np.zeros((dimt)); self.sigpref=np.zeros((self.dimt))
        self.lookup = np.clip(2/(inversefunc(log_mitsr_strain_rate,y_values=np.arange(-100,0,0.1),
                                                domain=0.00001,args=(self.beta2,self.Cc,self.Cr,self.CalphaNC))**beta2+1),0.0,1)*self.CalphaNC

        self.CalphaCc = self.CalphaNC/self.Cc

    def get_sigpref(self,sigma,e):
        sigpref=get_intersection(sigma,e,self.sigrangeNC,self.eNC,self.Cr)[0][0]
        return sigpref
        
    def initialize_reference_curve(self):
        self.sigrangeNC =  10**np.arange(-2,10,self.dsig)
        if self.ref_func=='loglog':
            self.rhoc=self.Cc/self.e0
            self.rhor=self.Cr/self.e0
            self.eNC = np.exp(np.log(self.e0) - self.rhoc*np.log(self.sigrangeNC))
        elif self.ref_func =='semilogx':
            self.eNC = self.e0 - self.Cc*np.log10(self.sigrangeNC)
                   
    def calc_pos_isotache(self,sigma,e,sigp,steps=200):
        sigp = np.clip(sigp,sigma,np.infty)
        OCR = np.clip((sigp)/(sigma),1,np.infty)
        if self.ref_func=='loglog':
            self.eUC0 = np.exp(np.log(self.e0) +(sigma<sigp)*(- self.rhoc*np.log(sigp) - (-self.rhor*np.log(sigp)+self.rhor*np.log(sigma))) - (sigma>=sigp)*self.rhoc*np.log(sigma))
        elif self.ref_func =='semilogx':
            self.eUC0 = self.e0 +(sigma<sigp)*(- self.Cc*np.log10(sigp) - (-self.Cr*np.log10(sigp)+self.Cr*np.log10(sigma))) - (sigma>=sigp)*self.Cc*np.log10(sigma)

        if self.isotache:
            eratei = mitsr_strain_rate(OCR,self.beta2,self.Cc,self.Cr,self.CalphaNC)*self.erateref
        else:
            eratei = beta3_fit(self.beta3,OCR)*self.erateref
        if self.use_power_law:
            Calphac = np.clip(power_law(np.clip(OCR,1.01,np.infty),self.m0,self.b0),0,1)*self.CalphaNC
        elif self.use_beta_nash:
            Calphac = np.clip(beta_nash(OCR,self.param_nash[0],self.param_nash[1],self.param_nash[2]),0,1)*self.CalphaNC #param_nash=[betamin,b0,a]
        else:
            if self.isotache:
                Calphac = np.clip(beta2_fit(self.beta2,OCR),0,1)*self.CalphaNC
            else:
                if self.beta2beta3:
                    Calphac = np.clip(2/((eratei/self.erateref)**(-self.beta2/self.beta3)+1),0.1,1)*self.CalphaNC
                else:
                    #Calphac = np.clip(fit_func.beta2_fit(beta2,OCR),0,1)*CalphaNC
                    Calphac = self.CalphaNC*Calpha_from_erate(np.array([eratei]),eref=self.erateref,Cc=self.Cc,Cr=self.Cr,
                                                Calpha=self.CalphaNC,beta2=self.beta2,beta3=self.beta3)[0]
                    #Calphac = np.interp(eratei/erateref,10**np.arange(-100,0,0.1),lookup)
                    
        deltae_target=e-self.eUC0
        eUC=self.eUC0
        eratei0=eratei
        for i in range(steps):
            deltae = -deltae_target/steps 
            eUC = eUC-deltae
            eratei = 10**(np.log10(eratei)-deltae/Calphac)
            #sigp =(sigp+deltae/(Cc-Cr))
            if self.Calpha_OCRref:
                if self.ref_func =='semilogx':
                    sigp = sigp+deltae/(self.Cc-self.Cr)*sigp
                elif self.ref_func == 'loglog':
                    sigp = sigp+np.log10((eUC+deltae)/eUC)/(self.rhoc-self.rhor)*sigp
                else:
                    sigp = get_intersection(np.array([sigma]),np.array([eUC]),self.sigrangeNC,self.eNC,self.Cr)[0][0]#
            #print(Calphac[1])
            OCR = sigp / sigma
            if self.use_power_law:
                Calphac = np.clip(power_law(np.clip(OCR,1.01,np.infty),self.m0,self.b0),0,1)*self.CalphaNC
            elif self.use_beta_nash:
                Calphac = np.clip(beta_nash(OCR,self.param_nash[0],self.param_nash[1],self.param_nash[2]),0,1)*self.CalphaNC #param_nash=[betamin,b0,a]
            else:
                if self.isotache:
                    Calphac = np.clip(beta2_fit(self.beta2,OCR),0,1)*self.CalphaNC
                else:
                    if self.beta2beta3:
                        Calphac = np.clip(2/((eratei/self.erateref)**(-self.beta2/self.beta3)+1),0.1,1)*self.CalphaNC
                    else:
                        #Calphac = np.clip(fit_func.beta2_fit(beta2,OCR),0,1)*CalphaNC
                        Calphac = self.CalphaNC*Calpha_from_erate(np.array([eratei]),eref=self.erateref,Cc=self.Cc,Cr=self.Cr,Calpha=self.CalphaNC,beta2=self.beta2,beta3=self.beta3)[0]
                        #Calphac = np.interp(eratei/erateref,10**np.arange(-100,0,0.1),lookup)
        return eratei

