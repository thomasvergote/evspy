def UCFit2(Calphahatc,Calphahats,erateci,eratesi,Cr,Cc,OCR_initial,OCR_final,dimt=100000,dtfactor=1e-7,Cv=5/3600/24/365,H=0.01,targettime=1e10,creep_coupled=False,sigmaref=100,beta2=4,beta3=19,hypA=False):
    # Used for AOS fitting
    # Adjust for use of ratio_Calphac_f_rate()
    CcCrRatio = Cc/Cr
    time=np.zeros((dimt))
    eratec=np.zeros((dimt)); erates=np.zeros((dimt)); erate_e=np.zeros((dimt))
    e=np.zeros((dimt)); erate=np.zeros((dimt)); ee=np.zeros((dimt))
    eratec[0]=erateci
    erates[0]=0#eratesi
    erate[0]=eratec[0]-erates[0]
    dt=1e-20#np.clip(1/np.abs(erate[0])*dtfactor,1e-5,1e-3)
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
                erates[t-1]=erates[t-1]*10**(5*np.log10((OCR_ref[t]-1)/(OCR_ref[t-1]-1))) # Viscous swelling develops and reduces at the same time while unloading ==> TO BE CHECKED EXPERIMENTALLY ON VARIOUS SCALES
                
            if t==1:
                erates[t-1]=eratesi*10**(5*np.log10(np.clip((OCR_ref[t]-1)/(OCR_final-1),0,np.infty)))
            if OCR_real[t]<1.05:
                erate_e[t]= Cr*0.434*(sigma[t]-sigma[t-1])/dt/sigma[t]
                if hypA:
                    erate_e[t]= Cc*0.434*(sigma[t]-sigma[t-1])/dt/sigma[t]
            else:
                erate_e[t]= Cr*0.434*(sigma[t]-sigma[t-1])/dt/sigma[t]
            ee[t]=ee[t-1]+erate_e[t]*dt #np.log10(OCR_real[t]) #
            e[t]=e[t-1]+(eratec[t-1]-erates[t-1])*dt+(ee[t]-ee[t-1])#(Cr*np.log10(1/OCR_real[t])-Cr*np.log10(1/OCR_real[t-1]))
            if creep_coupled==True:
                eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt #**2
                if OCR_final < OCR_initial: #13/01/2020: inclusion of loading 
                    if hypA:
                        eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt#+eratec[t-1]*erate_e[t]*(Cc-Cr)/Cr*1/(Calphahatc_real[t-1]*0.434)*dt #**2
                    else:
                        eratec[t]=eratec[t-1]-eratec[t-1]*(eratec[t-1]-erates[t-1])*1/(Calphahatc_real[t-1]*0.434)*dt+eratec[t-1]*erate_e[t]*(Cc-Cr)/Cr*1/(Calphahatc_real[t-1]*0.434)*dt #**2

            else:
                eratec[t]=eratec[t-1]-eratec[t-1]*eratec[t-1]*1/(Calphahatc_real[t-1]*0.434)*dt #**2
            erates[t]=erates[t-1]-erates[t-1]*erates[t-1]*1/(Calphahats_real[t-1]*0.434)*dt #**2
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
            t = 2#np.int(np.clip(t-2000,2,np.infty))
            failsafe += failsafe
            dtfactor = dtfactorinit / 10
            dtmult = 1+(dtmult-1) / 2
            dtfactorinit = dtfactorinit/10
            dt=1e-100#np.clip(1/np.abs(eratec[t]-erates[t])*dtfactor,1e-100,time[t]/2)
    if failsafe > 9:
        print(Calphahatc,Calphahats,erateci,eratesi,Cr)
    return time[:t],e[:t],erate[:t],eratec[:t],erates[:t],Calphahatc_real[:t],OCR_ref[:t],erate_e[:t]#,OCR_ref[:t]
