from os import listdir
from os.path import isfile, join
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputpath", dest="inputpath", default="", help="Input path")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="test.h5", help="Output h5 file")
args = parser.parse_args()

filelist = [join(args.inputpath, f) for f in listdir(args.inputpath) if isfile(join(args.inputpath, f)) and '.h5' in f]
allHDF = [pd.read_hdf(f, 'branches')  for f in filelist]
outputHDF = pd.concat(allHDF, ignore_index=True)
outputHDF.to_hdf('{}.h5'.format(outputfile.replace('.h5','')), 'branches', mode='a', format='table', append=True)



