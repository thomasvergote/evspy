# evspy
A repository on elasto-viscous-swelling-plastic modelling of soils. 

Currently includes the model used for http://doi.org/10.1680/jgeot.20.P.106. 

More features are planned in the future:
- a fully hydromechanically coupled model (details not yet published) with swelling and creep and for load and rate-controlled stress paths
- fitting module using MCMC
- documentation

## Data
Empirical data is added under `/data`. The files are HDF5 format and contain a range of stages with metadata. See the basic_examples to load the data. `loading_empirical_data.py` provides basic functions to load the data. 

![](example_fit.PNG)
