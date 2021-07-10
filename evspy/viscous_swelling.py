import numpy as np
from evspy.single_stage_model.creep_swelling_model import power_law

def swelling_law(erates,OCR,OCRrate,Calphahats_real,dt,CalphaNC,m1,b1,m2,b2,sigmarate,erate,reset):    
    OCRmin=1.05
    OCR=np.clip(OCR,OCRmin,np.infty)
    OCRrate=OCRrate*(OCR>OCRmin)
    
    try:
        min_erates=np.zeros((len(reset)))
        for x in range(len(reset)):
            if (reset[x]<2) and (OCR[x]>OCRmin).all() and ((sigmarate[x]<0) and (erate[x]>0)):
                #print(sigmarate[5])
                #print(erate[5])
                min_erates[x] = power_law(OCRmin,b=b2,m=m2)#*((sigmarate[x]<0)&(erate[x]>0))
                if x == 2:
                    Calphahats_real = np.clip(Calphahats_real,
                                                # Clip at Calphahat_s at OCR = 3 if load is changing and soil is straining to avoid high sensitivity during unloading
                                                CalphaNC*power_law(3,b=b1,m=m1)*((np.average(sigmarate)<-1e-20)&(np.average(erate)>1e-20)),np.infty)
                reset[x]+=1
                
            else:
                min_erates[x] = erates[x]/2
    except:
        if (reset<2) and (OCR>OCRmin).all():
            min_erates = power_law(OCRmin,b=b2,m=m2)*((np.average(sigmarate)<0)&(np.average(erate)>0))
            reset+=1
        else:
            min_erates = erates/2
    
    Calphahats_real = np.clip(Calphahats_real,CalphaNC*power_law(3,b=b1,m=m1)*((np.average(sigmarate)<-1e-20)&(np.average(erate)>1e-20)),np.infty)
    return np.clip(erates+m2*OCRrate/(OCR-1)*erates*dt-erates*erates/(Calphahats_real*0.434)*dt,min_erates,np.infty),Calphahats_real,reset