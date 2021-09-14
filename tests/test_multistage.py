from evspy.utils.loadsteps import LoadSteps
from evspy.multi_stage_model.decoupled_fdm import distorted_isotache_model
import numpy as np

def test_decoupled_consolidation_IL():
    load = 'test'
    load = LoadSteps()
    load.add_load_step(duration = 1e8,load = 5,rate = 1e-8,type_test = 'IL',
                        cv=7)
    model = distorted_isotache_model(load,sigma0=1,H=0.1,dimt=1000,e_init=1)
    model.initialize_decoupled()
    model.run_iterations()
    assert model.sigma[-1] == 5

def test_decoupled_consolidation_CRS():
    load=LoadSteps()
    load.add_load_step(duration = 1e8,
                        load = 5,
                        rate = 1e-8,
                        type_test = 'CRS',
                        cv=7)
    model = distorted_isotache_model(load,sigma0=1,H=0.1,dimt=1500,e_init=1)
    model.initialize_decoupled()
    model.run_iterations()
    assert np.round(model.sigma[-1],1) == 5
    assert np.round(np.log10(-model.erate[-1]),1) == -8