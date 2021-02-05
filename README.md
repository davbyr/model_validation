Model Validation Scripts
Author: David Byrne
Email: dbyrne@noc.ac.uk

These scripts are designed to analyse model data and compare it with observations. Each
script is designed to compare a specific piece of model data with a specific type of
observation. Additionally, each is separated into a 'Analysis' and
'Plotting' script. The analysis scripts expect inputs of model data and observations. They
will save their analysis to new netCDF files, which can be used by the plotting scripts
to make nice pictures.

All of the scripts are modular in nature, using Python definitions. This should make it
easier to modify specific bits of the script. Most importantly, each script has
routines for reading model data and observation data. By default, there are specific
datasets which the scripts are set up to read -- detailed more in the individual files.
You may wish to change these read functions. As long as the output from the read functions
follows the COAsT guidelines (see each file), the analysis and subsequently plotting
scripts will continue to work.

The following scripts use Python and the COAsT package. If the COAsT package is installed
to your Python environment, then other necessary packages will be too