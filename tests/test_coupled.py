from evspy.loadsteps import LoadSteps
from evspy.coupled_fdm import distorted_isotache_model
import numpy as np

def test_decoupled_consolidation_IL():
    load = 'test'
    load = LoadSteps()
    load.add_load_step(duration = 1e8,load = 5,rate = 1e-8,type_test = 'IL',
                        cv=7)
    model = distorted_isotache_model(load,sigma0=1,H=0.1,dimt=1000)
    model.initialize_decoupled(1,1,1e-10)
    model.run_iterations()
    assert model.sigma[-1] == 5

def test_decoupled_consolidation_CRS():
    load=LoadSteps()
    load.add_load_step(duration = 1e8,
                        load = 5,
                        rate = 1e-8,
                        type_test = 'CRS',
                        cv=7)
    model = distorted_isotache_model(load,sigma0=1,H=0.1,dimt=1500)
    model.initialize_decoupled(1,1,1e-10)
    model.run_iterations()
    assert np.round(model.sigma[-1],1) == 5
    assert np.round(np.log10(-model.erate[-1]),1) == -8
    

# def test_overall_coupled_CS_model():
#     load=LoadSteps()
#     load.add_load_step(1e5,5,1e-8,'CRS',cv=7)
#     load.add_load_step(1e5,15,1e-6,'CRS',cv=7)
#     load.add_load_step(1e5,30,1e-8,'CRS',cv=7)
#     load.add_load_step(1e5,70,1e-6,'CRS',cv=7)
#     load.add_load_step(1e5,100,1e-8,'CRS',cv=7)
#     load.add_load_step(1e5,200,1e-8,'IL',cv=7)
#     load.add_load_step(1e8,200,1e-8,'IL',cv=7)
#     load.add_load_step(1e8,20,-1e-6,'CRS',cv=7)
#     load.add_load_step(5e6,20,-1e-8,'IL',cv=7)
#     load.add_load_step(1e7,20,0,'CRS',cv=7)
#     model = distorted_isotache_model(load,sigma0=1,H=0.1)
#     # Add a check to see if initial condition is reasonable (for instance not above reference line)
#     model.initialize_decoupled(1,1,1e-10)
#     model.run_iterations()