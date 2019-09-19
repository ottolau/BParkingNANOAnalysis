# BParkingNANOAnalysis

This is an analyzer for analyzing the nanoAOD output of the BParkingNANO common framework. This package is written in python and can analyze the data parallelly. The whole framework uses uproot as ROOT files I/O, numpy and pandas for data selection.

## Getting started
To open nanoAOD files without trouble, we need a CMSSW environment.
```shell
cmsrel CMSSW_10_2_15
cd CMSSW_10_2_15/src
cmsenv
git cms-init
git clone https://github.com/ottolau/BParkingNANOAnalysis
. BParkingNANOAnalysis/BParkingNANOAnalyzer/setup.sh
scram b clean; scram b
```

## Run the analyzer

Inside the folder `python`, BaseAnalyzer.py contains the base class of the analyzer. The analyzer allows to output the results as tree (for further mva studies or unbinned fitting) or histograms. Any modification of the base class requires re-compiling the code.

The analyzers are put in the folder `scripts`. For example, BToKLLAnalyzer is the analyzer for B -> K ll decay channel. The analyzers for other channels should put in this folder as well.

The folder `test` is the working repository. To run the analyzer, do

```
python runBToKEEAnalyzer.py -i [inputFilesList] -o [outputFileName] -s -r
```

The descriptions of the options:
`-i`: A txt file contains the name of the nanoAOD files to be analyzed
`-o`: The name of the output file
`-s`: Select the output as histograms (By default, the output will be stored as tree)
`-r`: Enable multi-core processing

## Submit condor jobs

Inside the folder `test/condor`, there is an example script to submit condor jobs. Simply do

```
python submit_condor_lpc.py -i [inputFilesList]
```

For lxplus, do `voms-proxy-init -valid 192:00 -voms cms`, and copy the proxy `/tmp/x509up_XXXXX` to `$HOME/tmp/x509up`, then set `export X509_USER_PROXY=$HOME/tmp/x509up`. This allows us to use proxy in condor.


