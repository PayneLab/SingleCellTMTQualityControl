# SingleCellTMTQualityControl

This repository contains the methods and software used in the publication "Evaluating data quality in single cell proteomics - life at the signal/noise limit" by the Payne and Kelly labs at Brigham Young University.

## Publication
The application is currently in its initial stages and will be published when finished, both to bioRxiv as well as a peer-reviewed publication.

## Re-running these analyses
There are two options for reproducing the research reported in this repository. First, you can clone this repository onto your personal computer and run the notebooks that create the figures of the manuscript. Second, if you are looking for a no-install option, please follow this [link](https://mybinder.org/v2/gh/PayneLab/SingleCellTMTQualityControl/master) to a virtual machine hosting our code at Binder.

## Repository contents
This repository contains all information, data and code necessary to replicate the analyses in the manuscript. The file names are intentionally self explanatory, but are briefly reviewed below

* README.md - this introductory file
* data.md - contains basic information about where data files are located and what they mean
* data_descriptor.ipynb -  a tutorial jupyter notebook for how to access and use the data in this project
* ~/data - a folder that contains data files
* load_data.py - a python script containing all code required for parsing data files and loading them into data frames.
* plot_utils.py - a python script containing custom graphing functions
* make_figureA.ipynb - a jupyter notebook that contains all code used in making figure A for the manuscript
* make_figureB.ipynb - a jupyter notebook that contains all code used in making figure B for the manuscript
* make_figureC.ipynb - a jupyter notebook that contains all code used in making figure C for the manuscript
* requirements.txt - a list of packages that are needed for the Binder VM

## License
This package contains LICENSE.md document which describes the license for use. 

## Contact
This package is maintained by the Payne lab at Brigham Young University, https://payne.byu.edu
