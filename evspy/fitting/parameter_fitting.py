import multiprocessing
from IPython.display import clear_output
import lmfit

def param_def(Cc=1,Cr=0.2,Cv=5/3600/24/365,m1=0.8,b1=-0.8,m2=4,b2=-3.,beta2=3,beta3=18,Cr_vary=False,Cv_vary=False,CalphaNC=0.01,beta_vary=False):
    param=lmfit.Parameters()
    param.add('CalphaNC',value=CalphaNC,vary=False,min=0.00001);  
    param.add('m1',value=m1,vary=True,min=0.5,max=5);  
    param.add('b1',value=b1,vary=True,min=-8,max=0);  
    param.add('m2',value=m2,vary=True,min=0.5,max=5);
    param.add('b2',value=b2,vary=True,min=-8,max=-1);
    param.add('beta2',value=beta2,vary=beta_vary,min=0.01,max=10);
    param.add('beta3',value=beta3,vary=beta_vary,min=0,max=30);
    param.add('Cr',value=Cr,vary=Cr_vary,min=0.00001,max=0.5);                                             
    param.add('Cc',value=Cc,vary=False,min=0);                                              
    param.add('Cv',value=Cv,vary=Cv_vary,min=1/3600/24/365,max=10000/3600/24/365);                       
    return param

def residual_aos_overall(param,oed_data,fit_stages,load,stage_info,**kws):
    H0 = oed_data['sample_height'].dropna().iloc[0]
    time,sigma,e,erate,erate_e,erate_c,erate_s,Calphahats_real,ee,sigpref,time_step,OCR,OCRref,sigp_update=distorted_isotache_model(load,H=H0,plotting=False,**param,**kws)
    residual=np.array([])
    for stage in fit_stages:
        x,ysample,yfit=stage_selec(stage,stage_info,oed_data,time_step,time,sigma,e)
        yfit = yfit / np.abs(np.max(ysample)-np.min(ysample))
        ysample = ysample / np.abs(np.max(ysample)-np.min(ysample))
        residual=np.append(residual,((yfit-yfit[0])-(ysample-ysample[0])))
    print(np.sum(residual**2))
    return residual

def gradient_fit(param,oed_data,fit_stages,load,stage_info,method='lbfgsb',maxfev=10000,**kws):
    mi = lmfit.minimize(residual_aos_overall, param, args=(oed_data,fit_stages,load,stage_info),kws=kws, method=method,nan_policy='omit',options={'maxfev':maxfev}) #  
    lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
    H0 = oed_data['sample_height'].dropna().iloc[0]
    timeorg,sigmaorg,eorg,erate,erate_e,erate_c,erate_s,Calphahats_real,ee,sigpref,time_steporg,OCR,OCRref,sigp_update=distorted_isotache_model(load,H=H0,plotting=False,**param,**kws)
    time,sigma,e,erate,erate_e,erate_c,erate_s,Calphahats_real,ee,sigpref,time_step,OCR,OCRref,sigp_update=distorted_isotache_model(load,H=H0,plotting=False,**mi.params,**kws)
    for stage in fit_stages:
        plt.figure()
        x,ysample,yfit=stage_selec(stage,stage_info,oed_data,time_steporg,timeorg,sigmaorg,eorg)
        plt.semilogx(x,ysample,'o')
        plt.plot(x,yfit,'.-',label='original')
        x,ysample,yfit=stage_selec(stage,stage_info,oed_data,time_step,time,sigma,e)
        plt.plot(x,yfit,'.-',label='fitted')
        plt.ylim(np.min(ysample)/1.01,np.max(ysample)*1.01)
        plt.legend()
        plt.show()
    return mi

def model_in_dict(load1,res,param,kws,H=1,plotting=False):
    time,sigma,e,erate,erate_e,erate_c,erate_s,Calphahats_real,ee,sigpref,time_step,OCR,OCRref,sigp_update=distorted_isotache_model(load1,H=H,plotting=False,**param,**kws) 
    res['time']=time
    res['sigma']=sigma
    res['e']=e
    res['time_step']=time_step

def joint_residual_aos_overall(param,oed_data1,fit_stages1,oed_data2,fit_stages2,load1,load2,stage_info1,stage_info2,output=True,**kws):
    if output:
        print('m1='+str(param['m1'].value)+' b1='+str(param['b1'].value)+' m2='+str(param['m2'].value)+' b2='+str(param['b2'].value))
    load={0:load1,1:load2}
    H0={0:oed_data1['sample_height'].dropna().iloc[0],1:oed_data2['sample_height'].dropna().iloc[0]}
    
    #time,sigma,e,erate,erate_e,erate_c,erate_s,Calphahats_real,ee,sigpref,time_step,OCR,OCRref,sigp_update=distorted_isotache_model(load1,H=H0,plotting=False,**param,**kws) 
    #time,sigma,e,erate,erate_e,erate_c,erate_s,Calphahats_real,ee,sigpref,time_step,OCR,OCRref,sigp_update=distorted_isotache_model(load2,H=H0,plotting=False,**param,**kws)
    
    manager = multiprocessing.Manager()
    res = manager.dict()
    res[0]=manager.dict()
    res[1]=manager.dict()
    processes = [multiprocessing.Process(target=model_in_dict, args=(load[i],res[i],param,kws),
                                     kwargs={'H':H0[i]}) for i in [0,1]]
    [process.start() for process in processes] 
    [process.join() for process in processes]
    
    residual1=np.array([]); residual2=np.array([])
    if output:
        clear_output()
    for stage in fit_stages1:
        x,ysample,yfit=stage_selec(stage,stage_info1,oed_data1,res[0]['time_step'],res[0]['time'],res[0]['sigma'],res[0]['e'])
        yfit = yfit / np.abs(np.max(ysample)-np.min(ysample))
        ysample = ysample / np.abs(np.max(ysample)-np.min(ysample))
        if output:
            plt.semilogx(x,yfit)
            plt.plot(x,ysample,'o')
            plt.show()
        residual1=np.append(residual1,((yfit-yfit[0])-(ysample-ysample[0]))) 
    for stage in fit_stages2:
        x,ysample,yfit=stage_selec(stage,stage_info2,oed_data2,res[1]['time_step'],res[1]['time'],res[1]['sigma'],res[1]['e'])
        yfit = yfit / np.abs(np.max(ysample)-np.min(ysample))
        ysample = ysample / np.abs(np.max(ysample)-np.min(ysample))
        if output:
            plt.semilogx(x,yfit)
            plt.plot(x,ysample,'o')
            plt.show()
        residual2=np.append(residual2,((yfit-yfit[0])-(ysample-ysample[0]))) 
    residual=np.append(residual1,residual2)
    if output:
        print(np.str(np.sum(residual1**2))+' '+str(np.sum(residual2**2))+' '+str(np.sum(residual**2)))
    return residual

def joint_gradient_fit(param,oed_data1,fit_stages1,oed_data2,fit_stages2,load1,load2,stage_info1,stage_info2,method='lbfgsb',maxfev=10000,**kws):
    mi = lmfit.minimize(joint_residual_aos_overall, param, args=(oed_data1,fit_stages1,oed_data2,fit_stages2,load1,load2,stage_info1,stage_info2),kws=kws, method=method,nan_policy='omit',options={'maxfev':maxfev}) #  
    lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
    return mi

def joint_gradient_emcee(param,oed_data1,fit_stages1,oed_data2,fit_stages2,load1,load2,stage_info1,stage_info2,method='emcee',**kws):
    mi = lmfit.minimize(joint_residual_aos_overall, burn=10, steps=100, thin=2,params= param, args=(oed_data1,fit_stages1,oed_data2,fit_stages2,load1,load2,stage_info1,stage_info2),kws=kws, method=method,progress=True,nan_policy='omit') #  
    lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
    return mi

def ln_likelihood(p,x,y,stage,stage_info,**kws):
    H=stage_info[stage]['sample_height']
    resid = residual(param,oed_data,fit_stages,load,**kws)
    dy = np.exp(p['lnsigma'].value)
    return -0.5 * np.sum(((resid) / dy)**2 + np.log(2 * np.pi * dy**2))

def ln_prior(p,**kws):
    if np.array([(p[i].value>p[i].min)&(p[i].value<p[i].max) for i in p]).all():
        return 0
    return -np.inf

def ln_prob(p,x,y,stage,stage_info,log_scale,creep_coupled=True,swelling_isotache=True,beta2=4):
    if not np.isfinite(ln_prior(p,stage,stage_info,creep_coupled=creep_coupled,beta2=beta2)):
        return -np.inf
    return ln_likelihood(p,x,y,stage,stage_info,log_scale,creep_coupled=creep_coupled,swelling_isotache=swelling_isotache,beta2=beta2)+ ln_prior(p,stage,stage_info,creep_coupled=creep_coupled,beta2=beta2)

def corner_plot(res,stage,stage_info,test,creep_coupled=True,save=False):
    if res.params['Cr'].vary:
        fig,ax = plt.subplots(7,7,figsize=FormatFig.figsize(1,golden_mean=1))#=plt.figure(figsize=FormatFig.figsize(1))
    else:
        fig,ax = plt.subplots(5,5,figsize=FormatFig.figsize(1,golden_mean=1))#=plt.figure(figsize=FormatFig.figsize(1))
    samples = res.flatchain
    res2={}
    for i in res.params.keys():
        if res.params[i].vary:
            res2[i]=res.params[i].value
    if res.params['Cr'].vary:
        emcee_plot = corner.corner(samples.iloc[:,:], labels=[r'$\hat{C}_{\alpha,c}$', r'$\dot{e}_{c,init}$', r'$\hat{C}_{\alpha,s}$', r'$\dot{e}_{s,init}$',r'$C_r$',r'$C_v$', r'$ln(\sigma)$'],
                                   truths=list(res2.values()),bins=40,fig=fig)
    else:
        emcee_plot = corner.corner(samples.iloc[:,:], labels=[r'$\hat{C}_{\alpha,c}$', r'$\dot{e}_{c,init}$', r'$\hat{C}_{\alpha,s}$', r'$\dot{e}_{s,init}$', r'$ln(\sigma)$'],
                                   truths=list(res2.values()),bins=40,fig=fig)

def stage_selec(stage,stage_info,oed_data,time_step,time,sigma,e):
    plotdata=oed_data[oed_data['Stage Number']==stage]
    if (stage_info[stage]['type']=='CRS')|(stage_info[stage]['type']=='R'):
        x = 10**np.arange(np.log10(np.min(plotdata['Time since start of stage (s)'].iloc[1:])*1.02),np.log10(np.max(plotdata['Time since start of stage (s)'])*1.02),0.03)
        y = np.interp(np.log10(x),np.log10(np.clip(plotdata['Time since start of stage (s)'],0.1,np.infty)),plotdata['load'])
    else:
        x = 10**np.arange(np.log10(np.min(plotdata['Time since start of stage (s)'].iloc[1:])*1.02),np.log10(np.max(plotdata['Time since start of stage (s)'])*1.02),0.03)
        y = np.interp(np.log10(x),np.log10(np.clip(plotdata['Time since start of stage (s)'],0.1,np.infty)),plotdata['void_ratio']-plotdata['void_ratio'].iloc[0])
    try:
        load_step0=list(time_step)[stage-1]; load_step1=list(time_step)[stage]
        timefilter=(time>time_step[load_step0])&(time<=time_step[load_step1])
        if load_step1[:2]=='CR':
            yfit = np.interp(np.log10(x),np.log10(time[timefilter]-time_step[load_step0]),sigma[timefilter])
        else:
            yfit = np.interp(np.log10(x),np.log10(time[timefilter]-time_step[load_step0]),e[timefilter]-e[timefilter][0])  
    except:
        yfit = y*1000
    return x,y,yfit

def optim_minmaxrand(param,*args,mode='absolute',remark_1='',plotting=False,method_local='lbfgsb',remarks='',data_store='',**kws):
    method = 'optim_minmaxrand'
    global save_file
    save_file={}
    save_file[method]=pd.DataFrame(columns=[p for p in param if param[p].vary])
    
    comb={}
    out={}
    varyparam=[]
    for p in list(param):
        if param[p].vary==True:
            varyparam=varyparam+[p]
            comb[p]=[param[p].min+0.01]*(2**len(varyparam)-1)+[param[p].max-0.01]*(2**len(varyparam)-1)
            for k in varyparam[:-1]:
                comb[k]=comb[k]*2
    #last one first. max corner:
    min1=10;min2=50; min3=100
    min1_index=0; min2_index=1

    for p in varyparam:
        param[p].value=comb[p][-1]
    out[0] = gradient_fit(param,*args,method=method_local,**kws)
    i=1;save_file[method].loc[i+10]=0
    min1=np.sum((out[0].residual)**2)
    out_final = out[0]
    for p in varyparam:
        param[p].value=comb[p][0]
    out[1] = gradient_fit(param,*args,method=method_local,**kws)
    i=2; save_file[method].loc[i+10]=0
    if np.sum((out[1].residual)**2)<min1:
            min2=min1
            min1=out[1].residual
            out_final = out[1]
            min2_index=min1_index
            min1_index=1
    elif np.sum((out[1].residual)**2)<min2:
        min2=out[1].residual
        min2_index=1

    param_diff=[]
    for p in varyparam:
        param_diff=param_diff+[np.abs(out[min1_index].params[p]-out[min2_index].params[p])/np.abs(np.average([out[min1_index].params[p],out[min2_index].params[p]]))]

    while (((min2/min1)>1.5) or ((min3/min2)>1.5)) or (np.max(param_diff)>0.2):
        for p in varyparam:
            param[p].value=param[p].min+np.random.rand()*(param[p].max-param[p].min)
        out[i] = gradient_fit(param,*args,method=method_local,**kws)
        if remark_1=='':
            save_file_to_h5(save_file[method],data_store,method,dimension=mode,remarks=remarks)
        else:
            save_file_to_h5(save_file[method],data_store,method,dimension=remark_1,remarks=remarks)
        if np.sum((out[i].residual)**2)<min1:
            min3=min2
            min2=min1
            min1=np.sum((out[i].residual)**2)
            out_final = out[i]
            min2_index=min1_index
            min1_index=i
        elif np.sum((out[i].residual)**2)<min2:
            min3=min2
            min2=np.sum((out[i].residual)**2)
            min2_index=i
        elif np.sum((out[i].residual)**2)<min3:
            min3=np.sum((out[i].residual)**2)
        param_diff=[]
        for p in varyparam:
            param_diff=param_diff+[np.abs(out[min1_index].params[p]-out[min2_index].params[p])/np.abs(np.average([out[min1_index].params[p],out[min2_index].params[p]]))]

        print(i)
        i+=1
        print(str(min1)+'; '+str(min2)+'; '+str(min3)); print(np.max(param_diff))
    