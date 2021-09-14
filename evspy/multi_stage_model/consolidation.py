import numpy as np

def Consol_Terzaghi_Uavg_vertical(Cv,H,t=-1,targettime=1000,dimt=1000,time=-1,constant_dt_time=np.infty,dtmax=1e3):
    '''
    Parameters
    ----------
    Cv : float
        coefficient of vertical consolidation in m/s**2
    '''
    if dimt==-1:
        time=time
    else:
        if constant_dt_time>targettime:
            time=10**np.linspace(0,np.log10(targettime),dimt)
        else:
            dimt2 = np.int((targettime-constant_dt_time)/dtmax)
            time=np.append(10**np.linspace(0,np.log10(constant_dt_time),dimt),np.linspace(constant_dt_time+0.01,targettime,dimt2))
    if t>0:
        Tv=t*Cv/((H)**2)
    else:
        Tv=time*Cv/((H)**2)
    Tv=np.clip(Tv,0,10)
    UavgAnalV=np.sqrt(4*Tv/np.pi)/((1+(4*Tv/np.pi)**2.8)**0.179)
    return time,UavgAnalV