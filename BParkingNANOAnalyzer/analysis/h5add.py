from os import listdir
from os.path import isfile, join
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputpath", dest="inputpath", default="", help="Input path")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="test.h5", help="Output h5 file")
args = parser.parse_args()

ELECTRON_MASS = 0.000511
K_MASS = 0.493677
JPSI_LOW = 2.9
JPSI_UP = 3.3
B_LOWSB_LOW = 4.75
B_LOWSB_UP = 5.0
B_UPSB_LOW = 5.5
B_UPSB_UP = 5.75
B_MIN = 4.7
B_MAX = 6.0

allHDF = []
for f in [join(args.inputpath, f) for f in listdir(args.inputpath) if isfile(join(args.inputpath, f)) and '.h5' in f]:
  try:
    df = pd.read_hdf(f)
    df = df[(df['BToKEE_mll_raw'] > JPSI_LOW) & (df['BToKEE_mll_raw'] < JPSI_UP) & (df['BToKEE_l1_mvaId'] > 3.94) & (df['BToKEE_l2_mvaId'] > 3.94)]
    print(df.shape)
    allHDF.append(df)
  except ValueError:
    print('Empty HDF file')
if len(allHDF) != 0:
  outputHDF = pd.concat(allHDF, ignore_index=True)
else:
  outputHDF = pd.DataFrame()

print(outputHDF.shape)

#allHDF = [pd.read_hdf(f, 'branches')  for f in filelist]
#outputHDF = pd.concat(allHDF, ignore_index=True)
outputHDF.to_hdf('{}.h5'.format(args.outputfile.replace('.h5','')), 'branches', mode='a', format='table', append=True)



