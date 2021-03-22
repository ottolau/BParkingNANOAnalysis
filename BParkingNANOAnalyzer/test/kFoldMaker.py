import ROOT
import pandas as pd
import os
import multiprocessing as mp
from functools import partial
import sys
sys.path.append('../')
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from operator import itemgetter 

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfiles", dest="inputfiles", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
args = parser.parse_args()


ROOT.gROOT.SetBatch()

if __name__ == "__main__":
    if not '.list' in args.inputfiles:
      print('Not a .list file')
      sys.exit()

    with open(args.inputfiles) as filenames:
        fileList = [f.rstrip('\n') for f in filenames]


    print(len(fileList))
    cv = KFold(n_splits=5, shuffle=True)
    fold = 1
    for train_idx, test_idx in cv.split(fileList):
      file_train = list(itemgetter(*train_idx)(fileList))
      file_test = list(itemgetter(*test_idx)(fileList))
      outputFile_train = open('{}_fold_{}_training.list'.format(args.inputfiles.replace('.list',''), fold), 'w+')
      for f in file_train:
          outputFile_train.write('%s\n'%(f))
      outputFile_train.close()
      outputFile_test = open('{}_fold_{}_testing.list'.format(args.inputfiles.replace('.list',''), fold), 'w+')
      for f in file_test:
          outputFile_test.write('%s\n'%(f))
      outputFile_test.close()
      fold += 1


