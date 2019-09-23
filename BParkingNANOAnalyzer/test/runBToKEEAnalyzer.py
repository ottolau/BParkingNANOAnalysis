import ROOT
import pandas as pd
import os
import multiprocessing as mp
import sys
sys.path.append('../')
from scripts.BToKLLAnalyzer import BToKLLAnalyzer

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfiles", dest="inputfiles", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-m", "--maxevents", dest="maxevents", type=int, default=ROOT.TTree.kMaxEntries, help="Maximum number events to loop over")
parser.add_argument("-t", "--ttree", dest="ttree", default="Events", help="TTree Name")
parser.add_argument("-s", "--hist", dest="hist", action='store_true', help="Store histograms or tree")
parser.add_argument("-r", "--runparallel", dest="runparallel", action='store_true', help="Enable parallel run")
args = parser.parse_args()


ROOT.gROOT.SetBatch()

def exec_me(command, dryRun=False):
    print(command)
    if not dryRun:
        os.system(command)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    n = max(1, n)
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def analyze(inputfile, outputfile, hist=False):
    analyzer = BToKLLAnalyzer(inputfile, outputfile, hist)
    analyzer.run()

def analyzeParallel(enumfChunk):
    ich, fChunk = enumfChunk
    print("Processing chunk number %i"%(ich))
    outputfile = outpath+'/'+args.outputfile.replace('.root','')+'_subset'+str(ich)+'.root'
    analyze(fChunk, outputfile, args.hist)


if __name__ == "__main__":
    if not args.runparallel:
        with open(args.inputfiles) as filenames:
            fileList = [f.rstrip('\n') for f in filenames]
        inputfile = fileList
        outputfile = args.outputfile.replace('.root','')+'.root'
        analyze(inputfile, outputfile, args.hist)

    else:
        #outputBase = "/eos/uscms/store/user/klau/BsPhiLL_output/LowPtElectronSculpting"
        #outputFolder = "BsPhiEE_CutBasedEvaluation"
        global outpath
        #outpath  = "%s/%s"%(outputBase,outputFolder)
        outpath = '.'
        if not os.path.exists(outpath):
            exec_me("mkdir -p %s"%(outpath), False)

        with open(args.inputfiles) as filenames:
            fileList = [f.rstrip('\n') for f in filenames]
        group  = 6
        # stplie files in to n(group) of chunks
        fChunks= list(chunks(fileList,group))
        print ("writing %s jobs"%(len(fChunks)))

        pool = mp.Pool(processes = 4)
        input_parallel = list(enumerate(fChunks))
        print(input_parallel)
        pool.map(analyzeParallel, input_parallel)
        pool.close()
        pool.join()

        outputfile = args.outputfile.replace('.root','')
        if args.hist:
          exec_me("hadd -k -f %s/%s %s/%s"%(outpath,outputfile+'.root',outpath,outputfile+'_subset*.root'))
          exec_me("rm %s/%s"%(outpath,outputfile+'_subset*.root'))

        else:
          allHDF = [pd.read_hdf(f, 'branches')  for f in ['{}/{}'.format(outpath, outputfile+'_subset{}.h5'.format(i)) for i in range(len(fChunks))]]
          outputHDF = pd.concat(allHDF, ignore_index=True)
          outputHDF.to_hdf('{}/{}.h5'.format(outpath, outputfile), 'branches', mode='a', format='table', append=True)
          exec_me("rm %s/%s"%(outpath,outputfile+'_subset*.h5'))





