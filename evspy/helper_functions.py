import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pynverse import inversefunc
import lmfit
from scipy.special import lambertw
from scipy.optimize import minimize

# Intersections

def semilogintersection(e0,e1,sigma1,Cc,Cr):
    '''Intersection between two compression curves using semilogarithmic relations

    Parameters
    ----------
    e0 : float
        reference void ratio (y-axis) of compression curve
    e1 : float
        void ratio (y-axis) at current stress point
    sigma1 : float
        stress (x-axis) at current stress point
    Cc : float
        compression index
    Cr : float
        recompression index

    Returns
    -------
    tuple of floats
        coordinates at intersection
    '''
    m1=-Cc
    m2=-Cr
    b1=e0
    b2=e1+Cr*np.log10(sigma1)
    xi = (b1-b2) / (m2-m1)
    yi = m1 * xi + b1
    xi = 10**xi
    return xi,yi
semilogintersection=np.vectorize(semilogintersection)

def loglogintersection(e0,e1,sigma1,rhoc,Cr):
    '''Intersection between two compression curves using log-log relations

    Parameters
    ----------
    e0 : float
        reference void ratio (y-axis) of compression curve
    e1 : float
        void ratio (y-axis) at current stress point
    sigma1 : float
        stress (x-axis) at current stress point
    rhoc : float
        log-log compression index
    Cr : float
        recompression index
    
    Returns
    -------
    float
        x-coordinate at intersection
    '''
    def f(sigma,rhoc,Cr,e0,e1,sigma1):
        return np.abs(np.exp(np.log(e0)-rhoc*np.log(sigma))-e1+Cr*np.log10(sigma/sigma1))
    return minimize(f,x0=sigma1,args=(rhoc,Cr,e0,e1,sigma1))['x']

def line_intersection(line1, line2):
    '''Finds intersection between two lines given end points of the lines

    Parameters
    ----------
    line1 : tuple
        ((x0,y0),(x1,y1)) coordinates of end points of first line
    line2 : tuple
        (x0,y0),(x1,y1)) coordinates of end points of second line

    Returns
    -------
    tuple
        intersection point
    '''
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    #if div == 0:
       #raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_intersection(sigrangeUC,eUC,sigrangeNC,eNC,Cr):
    y=(eUC-Cr*np.log10(sigrangeNC[:,None]/sigrangeUC[None,:]))
    idx=pd.DataFrame(np.argwhere(np.diff(np.sign(y.transpose() - eNC)))).drop_duplicates(subset=[0])
    idx.index=idx[0]
    df=pd.DataFrame()
    df['target']=np.linspace(0,len(eUC)-1,len(eUC))
    idx=pd.concat([df,idx],axis=1).interpolate()
    idx=idx.drop('target',axis=1).values
    idx=idx.astype('int')
    return sigrangeNC[idx[:,1]], eNC[idx[:,1]]

def make_grid(df,sigrange,de=0.0002):
    sigi=sigrange
    ei=10**np.arange(np.log10(0.5),np.log10(3),de)
    erateii=griddata((df['sigma'],df['e']),np.log10(df['erate']),xi=(sigi[None,:],ei[:,None]))
    return sigi,ei,erateii