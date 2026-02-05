# SHARKpy
This repository provides the code and tools to apply a dual-station approach for volcano-seismic monitoring. A continuos activity index, SHARK, is computed. Further, magnitude estimation is supported.
## Data
Set the data path in class kava_ini. Data should be provided as mseed-files. One file for each component. Only vertical component is required, horizontal components are optional.
The path strucure should be organised as follows:
"rootdata/stationdir/stationid+yymmddhhmmss+.pri0" and "+.pri1","+.pri2" respectively.

## Input parameters
Custom input parameter are defined in the ini class kava_ini

## The packages
### kavutil_2
Various functions for the computation of SHARK.

### magnitude_estimation
Function to remove the instrument response from the recorded data traces. Poles and Zeros for two commonly used seismometers are provided in kav_init.py. 
Implementation for the calculation of the local vertical magnitude follows Bakun and Joyner (1984) (DOI:10.1785/BSSA0740051827)

### compute_frequencybands
This package is used to compute the dominant frequency bands for each seismic station using a Kernel Density Estimation approach.

### runplotcatalog1_6
This packages provides the pipeline for the computation of the activity index, SHARK, and the event detection.

### plutil
Contains the plotting routines for visualisation of the index and its detections.
