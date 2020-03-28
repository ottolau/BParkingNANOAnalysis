import sys, os, fnmatch
import ROOT
from os import listdir
from os.path import isfile, join

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputpath", dest="inputpath", default="DoubleMuonNtu_Run2016B.list", help="Input path")
parser.add_argument("-o", "--outputpath", dest="outputpath", default="plots.root", help="Output path")
args = parser.parse_args()

def exec_me(command, dryRun=False):
    print(command)
    if not dryRun:
        os.system(command)

def missing_elements(L):
  start, end = L[0], L[-1]
  return sorted(set(range(start, end + 1)).difference(L))

if __name__ == '__main__':
    dryRun  = False
    subdir  = os.path.expandvars("$PWD")

    inputpath = args.inputpath
    outputpath = args.outputpath

    filenumber = sorted([int(f.replace('.root','').replace('subset','').split('_')[-1]) for f in listdir(outputpath) if isfile(join(outputpath, f)) and '.root' in f])
    missing = missing_elements(filenumber)
    print("Missing: {}".format(missing))

    for i in missing:
        inpath  = "%s/sub_%d/"%(inputpath,i)
        os.chdir(os.path.join(subdir,inpath))
        print(os.getcwd())
        job_name = "runjob_%s.jdl"%i
        if not dryRun:
            os.system("condor_submit %s"%job_name)
        os.chdir(subdir)


