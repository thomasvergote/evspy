# evspy - 1D elasto-viscoplastic modelling of soils with swelling
A repository on elasto-viscous-swelling-plastic modelling of soils. 

![github-actions](https://github.com/thomasvergote/evspy/actions/workflows/python-test.yml/badge.svg)

Two models are incorporated:
- a decoupled model to model and interpret laboratory test data on soil after unloading: http://doi.org/10.1680/jgeot.20.P.106. 
- a coupled model that incorporates the features inferred from the decoupled model: http://dx.doi.org/10.1002/nag.3248

More features are planned in the future:
- fitting module using MCMC
- documentation

## Data
Empirical data is added under `/data`. The files are HDF5 format and contain a range of stages with metadata (such as the OCR of the test stage). See the basic_examples to load the data. `loading_empirical_data.py` provides basic functions to load the data. 

![example](/example_fit.png)
