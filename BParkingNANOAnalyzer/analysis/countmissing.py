from os import listdir
from os.path import isfile, join
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputpath", dest="inputpath", default="", help="Input path")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="test.h5", help="Output h5 file")
args = parser.parse_args()

def missing_elements(L):
  start, end = L[0], L[-1]
  return sorted(set(range(start, end + 1)).difference(L))

filenumber = sorted([int(f.replace('.h5','').replace('subset','').split('_')[-1]) for f in listdir(args.inputpath) if isfile(join(args.inputpath, f)) and '.h5' in f])
print(missing_elements(filenumber))




