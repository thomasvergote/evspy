def UCFit3(Calphahatc,Calphahats,erateci,eratesi,Cr,OCR,Cc,dimt=100000,dtfactor=1e-7,Cv=5/3600/24/365,H=0.01,targettime=1e10,creep_coupled=False,sigmaref=100,beta2=3,beta3=19):
    #UCFit3 was conceived as part of article 2 reasoning: should Calpha,c be a function of OCR, or is it really a function of e_c and is e_c a function of OCR?
    # Use of ratio_Calphac_f_rate()
    with np.errstate(divide='ignore', invalid='ignore'):
        CcCrRatio=Cc/Cr
        dtfactorinit=dtfactor
        time=np.zeros((dimt))
        eratec=np.zeros((dimt)); erates=np.zeros((dimt))
        e=np.zeros((dimt)); erate=np.zeros((dimt)); ee=np.zeros((dimt))
        eratec[0]=erateci
        erates[0]=0#eratesi
        erate[0]=eratec[0]-erates[0]
        dt=1e-20#np.clip(1/np.abs(erate[0])*dtfactor,1e-5,1e-3)
        OCR_real=np.zeros((dimt))
        OCR_real[0]=1
        sigp_Calpha=np.zeros((dimt))
        sigp_Calpha[0]=sigmaref
        Calphahatc_real=np.zeros((dimt))
        t=0
        erateref=erateci/beta3_fit(beta3,OCR)
        if creep_coupled==True:
            Calphahatc_real[0]=Calphahatc#ratio_Calphacinit(1,beta2)*(Calphahatc/ratio_Calphacinit(OCR,beta2))
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
                    #print(np.log10((OCR_real[t]-1)/(OCR_real[t-1]-1))
                    erates[t-1]=erates[t-1]*10**(5*np.log10(np.clip(OCR_real[t]-1,0.0000000000001,np.infty)/(OCR_real[t-1]-1))) # Viscous swelling develops and reduces at the same time while unloading ==> TO BE CHECKED EXPERIMENTALLY ON VARIOUS SCALES
                if t==1:
                    erates[t-1]=eratesi*10**(5*np.log10((OCR_real[t]-1)/(OCR-1)))
                ee[t]=-Cr*np.log10(OCR)*UavgAnal #np.log10(OCR_real[t]) #Strictly more correct, but not such a good fit. Stick to 'naive' expression (in UCFit2, for OCRinitial=1, same as the "correct" expression)
                e[t]=e[t-1]+(eratec[t-1]-erates[t-1])*dt+(ee[t]-ee[t-1])#(Cr*np.log10(1/OCR_real[t])-Cr*np.log10(1/OCR_real[t-1]))
                if creep_coupled==True:
                    eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt #**2
                else:
                    eratec[t]=eratec[t-1]-eratec[t-1]*eratec[t-1]*1/(Calphahatc_real[t-1]*0.434)*dt #**2
                erates[t]=erates[t-1]-erates[t-1]*erates[t-1]*1/(Calphahats*0.434)*dt #**2
                erate[t]=(e[t]-e[t-1])/dt#
                sigp_Calpha[t]=sigp_Calpha[t-1]+(OCR_real[t]-OCR_real[t-1])*sigmaref
                sigp_Calpha[t]=sigp_Calpha[t]+sigp_Calpha[t]*(eratec[t]-erates[t])/(Cr*CcCrRatio-Cr)/0.434*dt

                if creep_coupled==True:
                    Calphahatc_real[t]=Calphahatc*ratio_Calphac_f_rate(eratec[t]/erateref,beta2,beta3)/ratio_Calphac_f_rate(erateci/erateref,beta2,beta3)#
                else:
                    Calphahatc_real[t]=Calphahatc
                dt=np.clip(1/np.abs(eratec[t]-erates[t])*dtfactor,1e-100,time[t]/2)
            else:
                t = 2#np.int(np.clip(t-2000,2,np.infty))
                failsafe += failsafe
                dtfactor = dtfactorinit / 10
                dtmult = 1+(dtmult-1) / 2
                dtfactorinit = dtfactorinit/10
                dt=1e-100#np.clip(1/np.abs(eratec[t]-erates[t])*dtfactor,1e-100,time[t]/2)
        if failsafe > 9:
            print(Calphahatc,Calphahats,erateci,eratesi,Cr)
        return time[:t],e[:t],erate[:t],eratec[:t],erates[:t],Calphahatc_real[:t],OCR_real[:t]
