import datetime
import pandas as pd
import numpy as np
from evspy.model_run import decoupled_cs_model

def test_decoupled_CS_model():
    Calphahatc = 0.0015
    Calphahats = 0.0005
    erateci = 1e-8
    eratesi = 1e-6
    Cr = 0.05
    Cc = Cr*5
    CalphaNC = Cc * 0.04
    Cv = 2/365/3600
    OCR_initial = 1.0
    OCR_final = 1.5
    model = decoupled_cs_model(Calphahatc, Calphahats,  erateci, eratesi, Cc,Cr,CalphaNC, OCR_initial,OCR_final,Cv=Cv,nonlinear_Calphahat=True,hypA=False,beta2=4,beta3=20)
    
    assert np.round(model.OCR[-1],2) == OCR_final