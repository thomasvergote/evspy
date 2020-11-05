import pandas as pd

def h5store(filename, df, tst, **kwargs):
    store = pd.HDFStore(filename)
    store.put(tst, df)
    store.get_storer(tst).attrs.metadata = kwargs
    store.close()

def h5load(store,tst):
    data = store[tst]
    metadata = store.get_storer(tst).attrs.metadata
    return data, metadata

def import_data(data_store='../data/Empirical_data_literature'):
    readings={}; metadata_readings={}
    with pd.HDFStore(data_store) as hdf:
        test=hdf.keys()
        for tst in test:
            readings[tst], metadata_readings[tst] = h5load(hdf,tst)
    metadata_readings=pd.DataFrame(metadata_readings).transpose()
    for i in metadata_readings.index:
        if '/AN' in i:
            metadata_readings.loc[i,'CalphaNC']=0.008
    metadata_readings.loc['/AN-1.05_1.csv','OCR']=1.4578
    metadata_readings.loc['/AN-1.05_2.csv','OCR']=1.4526
    metadata_readings.loc['/AN-1.05_3.csv','OCR']=1.053
    metadata_readings.loc['/AN-1.2_1.csv','OCR']=1.553
    metadata_readings.loc['/AN-1.2_2.csv','OCR']=1.49
    metadata_readings.loc['/AN-1.2_3.csv','OCR']=1.2
    metadata_readings.loc['/AN-1.2_1.csv','OCR']=2.59
    metadata_readings.loc['/AN-2_2.csv','OCR']=2.48
    metadata_readings.loc['/AN-2_3.csv','OCR']=2

    F_B_list= metadata_readings[metadata_readings['dataset']=='Feng (1991), Berthierville clay'].index
    T_list  = metadata_readings[metadata_readings['dataset']=='Tanaka et al. (2014)'].index #['/TA-1.14_.csv','/TA-2.56_.csv','/TA-3.00_.csv']# 
    A_list  = metadata_readings[metadata_readings['dataset']=='Alonso and Navarro (2005)'].index
    F_H_list=metadata_readings[metadata_readings['dataset']=='Feng (1991), Saint Hilaire clay'].index
    F_V_list = metadata_readings[metadata_readings['dataset']=='Feng (1991), Vasby clay'].index
    S_list = ['/S_OCR 1.5','/S_OCR 3.0','/S_OCR 6.5']
    F_list= ['/F_OCR=1.5','/F_OCR = 2.0', '/F_OCR = 4.0', '/F_OCR = 6.0', '/F_OCR = 8.0'] #'/F_OCR=1.5','/F_OCR = 2.0', '/F_OCR = 4.0', '/F_OCR = 6.0', '/F_OCR = 8.0'
    testlist = {'F_V':F_V_list,'F_B':F_B_list,'F_H':F_H_list,'TA':T_list,'AN':A_list,'S':S_list,'F':F_list}

    Cr_current={'/AN-1.05_1.csv': 0.0005, '/AN-1.05_2.csv': 0.0005, '/AN-1.05_3.csv': 0.0005, '/AN-1.2_1.csv': 0.0005, '/AN-1.2_2.csv': 0.0005, '/AN-1.2_3.csv': 0.0005, '/AN-2_1.csv': 0.0005, '/AN-2_2.csv': 0.0005, '/AN-2_3.csv': 0.0005,'/S_OCR 1.5': 0.012, '/S_OCR 3.0': 0.02, '/S_OCR 6.5': 0.035,'/V-acc_0.4': 0.04, '/V-b2_0.2': 0.035, '/V-b2_1.0': 0.045, '/V-b3_0.25': 0.03, '/V-b4_0.5': 0.029, '/V-b4_2.54': 0.046, '/V-b5_1.0': 0.035, '/V-b6_0.5': 0.031, '/V-b6_17.0': 0.055, '/V-b8_0.1': 0.03, '/V-b8_0.25': 0.03,'/TA-3.00_.csv': 0.043, '/TA-2.56_.csv': 0.045,'/TA-1.14_.csv': 0.015, '/TA-1.27_.csv': 0.028, '/TA-1.49_.csv': 0.032, '/TA-2.00_.csv': 0.037,'/SH-03_0.2_1': 0.015, '/SH-03_0.2_2': 0.015, '/SH-03_0.2_3': 0.018, '/SH-04_0.4': 0.023, '/SH-06_1.0': 0.03, '/SH-07_0.6': 0.02, '/SH-08_1.0': 0.02, '/SH-12_1.5': 0.04, '/SH-12_3.25': 0.045, '/SH-18_0.1': 0.035, '/SH-18_0.2': 0.025, '/SH-b1_2.2': 0.045,'/MF-0.14_.csv': 0.005, '/MF-0.1_.csv': 0.005, '/MF-0.25_.csv': 0.005, '/MF-0.2_.csv': 0.005, '/MF-0.2_2.csv': 0.005, '/MF-0.33_.csv': 0.008, '/MF-0.4_.csv': 0.008, '/MF-0.5_.csv': 0.006, '/MF-0.5_2.csv': 0.0055, '/MF-0.6_.csv': 0.008, '/MF-0.75_.csv': 0.01, '/MF-15_.csv': 0.009, '/MF-1_.csv': 0.009, '/MF-23_.csv': 0.012, '/MF-3_.csv': 0.008, '/MF-4_.csv': 0.008, '/MF-7_.csv': 0.0095,'/F_OCR=1.5':0.06,'/F_OCR = 2.0':0.07, '/F_OCR = 4.0':0.095, '/F_OCR = 6.0':0.095, '/F_OCR = 8.0':0.095}
    Cv_current={'/AN-1.05_1.csv': 6.341958396752918e-07, '/AN-1.05_2.csv': 6.341958396752918e-07, '/AN-1.05_3.csv': 6.341958396752918e-07, '/AN-1.2_1.csv': 6.341958396752918e-07, '/AN-1.2_2.csv': 6.341958396752918e-07, '/AN-1.2_3.csv': 6.341958396752918e-07, '/AN-2_1.csv': 6.341958396752918e-07, '/AN-2_2.csv': 6.341958396752918e-07, '/AN-2_3.csv': 6.341958396752918e-07,'/S_OCR 1.5': 9.512937595129376e-08, '/S_OCR 3.0': 1.5854895991882294e-07, '/S_OCR 6.5': 9.512937595129376e-08,'/V-acc_0.4': 3.8051750380517503e-07, '/V-b2_0.2': 3.8051750380517503e-07, '/V-b2_1.0': 9.512937595129376e-08, '/V-b3_0.25': 9.512937595129376e-08, '/V-b4_0.5': 3.170979198376459e-07, '/V-b4_2.54': 3.8051750380517503e-07, '/V-b5_1.0': 2.219685438863521e-07, '/V-b6_0.5': 4.7564687975646883e-07, '/V-b6_17.0': 1.9025875190258752e-07, '/V-b8_0.1': 1.9025875190258752e-07, '/V-b8_0.25': 1.9025875190258752e-07,'/TA-3.00_.csv': 0.5098427194317604e-06, '/TA-2.56_.csv': 0.6098427194317604e-06, '/TA-1.14_.csv': 1.1098427194317604e-06, '/TA-1.27_.csv': 1.1098427194317604e-06, '/TA-1.49_.csv': 1.1098427194317604e-06, '/TA-2.00_.csv': 0.6098427194317604e-06,'/SH-03_0.2_1': 1.1098427194317604e-06, '/SH-03_0.2_2': 1.1098427194317604e-06, '/SH-03_0.2_3': 1.744038559107052e-06, '/SH-04_0.4': 6.341958396752918e-07, '/SH-06_1.0': 7.927447995941147e-07, '/SH-07_0.6': 6.341958396752918e-07, '/SH-08_1.0': 4.7564687975646883e-07, '/SH-12_1.5': 4.7564687975646883e-07, '/SH-12_3.25': 3.8051750380517503e-07, '/SH-18_0.1': 3.8051750380517503e-07, '/SH-18_0.2': 7.927447995941147e-07, '/SH-b1_2.2': 6.341958396752918e-07,'/MF-0.14_.csv': 3.170979198376459e-07, '/MF-0.1_.csv': 3.170979198376459e-07, '/MF-0.25_.csv': 3.170979198376459e-07, '/MF-0.2_.csv': 3.170979198376459e-07, '/MF-0.2_2.csv': 3.170979198376459e-07, '/MF-0.33_.csv': 9.512937595129377e-07, '/MF-0.4_.csv': 9.512937595129377e-07, '/MF-0.5_.csv': 4.7564687975646883e-07, '/MF-0.5_2.csv': 6.341958396752918e-07, '/MF-0.6_.csv': 6.341958396752918e-07, '/MF-0.75_.csv': 9.512937595129377e-07, '/MF-15_.csv': 9.512937595129377e-07, '/MF-1_.csv': 9.512937595129377e-07, '/MF-23_.csv': 7.512937595129377e-07, '/MF-3_.csv': 9.512937595129377e-07, '/MF-4_.csv': 9.512937595129377e-07, '/MF-7_.csv': 9.512937595129377e-07,'/F_OCR=1.5':2.0e-07,'/F_OCR = 2.0':2.0e-07, '/F_OCR = 4.0':0.9e-07, '/F_OCR = 6.0':1.0e-07, '/F_OCR = 8.0':1.0e-07}
    CalphaCc={'F_B':0.04,'TA':0.04,'F_H':0.04,'F_V':0.06,'S':0.06,'AN':0.04,'F':0.04}
    return readings,metadata_readings,Cr_current,Cv_current,CalphaCc,testlist