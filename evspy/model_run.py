import evspy.creep_swelling_model as cs

class decoupled_cs_model:
    def __init__(self, Calphahatc, Calphahats,  erateci, eratesi, Cc, Cr, CalphaNC, OCR_initial,OCR_final, sigma0 = 100, sigmap = 100,
                 dimt=100000,dtfactor=1e-7,Cv=5/3600/24/365,H=0.01,targettime=1e10, swelling_isotache = False,
                 nonlinear_Calphahat=True,sigmaref=100,beta2=4,beta3=19,hypA=False,
                model_type = 'load_controlled'):
        self.Calphahatc = Calphahatc
        self.Calphahats = Calphahats
        self.CalphaNC = CalphaNC
        self.erateci = erateci
        self.eratesi = eratesi
        self.Cc = Cc
        self.Cr = Cr
        self.OCR_initial = OCR_initial
        self.OCR_final = OCR_final
        self.dimt = dimt
        self.dtfactor = dtfactor
        self.Cv = Cv
        self.H = H
        self.targettime = targettime
        self.nonlinear_Calphahat = nonlinear_Calphahat
        self.sigmaref = sigmaref
        self.beta2 = beta2
        self.beta3 = beta3
        self.hypA = hypA
        self.model_type = model_type
        self.g = 9.81
        self.sigma0 = sigma0
        self.swelling_isotache = swelling_isotache
        self.sigmap = sigmap
        
        if model_type == 'load_controlled':
            res = cs.C_S_model_with_load(self.Calphahatc, self.Calphahats,  self.erateci, self.eratesi, self.Cr,self.Cc, 
                                        self.OCR_initial,self.OCR_final,dimt=self.dimt,dtfactor = self.dtfactor,Cv = self.Cv,H = self.H,targettime = self.targettime,
                                            nonlinear_Calphahat = self.nonlinear_Calphahat, sigmaref = self.sigmaref,beta2 = self.beta2,beta3 = self.beta3,hypA = self.hypA)
        elif model_type == 'relaxation':
            res = cs.C_S_relaxation(self.Calphahatc, self.Calphahats,  self.erateci, self.eratesi, self.Cr,self.Cc,self.sigma0,dimt=self.dimt,dtfactor = self.dtfactor,
                                             targettime = self.targettime,nonlinear_Calphahat = self.nonlinear_Calphahat, swelling_isotache=self.swelling_isotache,
                                             OCR = self.OCR_final, beta2 = self.beta2, beta3 = self.beta3)
        elif model_type == 'rate_controlled':
            res = cs.C_S_CRS(self.Cc,self.Cr,self.sigmap,self.erateci,self.CalphaNC,erateref = self.erateref, Cc_reduction=2.3,
                                Cc_f_OCR=False,erates0=-1,erate = 1,sigma0=265,sigmaf=160,b2=-5.5,m2=2.5,beta2=self.beta2,beta3=self.beta3,estart=1,m1=0.88,b1=-1.4,dtfactor=1)

        self.time = res['time']
        self.e = res['e']
        self.erate = res['erate']
        self.erate_c = res['erate_c']
        self.erate_s = res['erate_s']
        self.Calphahatc = res['Calphahatc']
        if model_type == 'load_controlled':
            self.OCR = res['OCR']
            self.OCR_ref = res['OCR_ref']
            self.erate_e = res['erate_e']
        if model_type == 'relaxation':
            self.sigma = res['sigma']
        
