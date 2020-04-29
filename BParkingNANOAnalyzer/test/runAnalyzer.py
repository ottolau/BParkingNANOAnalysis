import ROOT
import pandas as pd
import os
import multiprocessing as mp
from functools import partial
import sys
sys.path.append('../')

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfiles", dest="inputfiles", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-m", "--maxevents", dest="maxevents", type=int, default=ROOT.TTree.kMaxEntries, help="Maximum number events to loop over")
parser.add_argument("-t", "--ttree", dest="ttree", default="Events", help="TTree Name")
parser.add_argument("-s", "--hist", dest="hist", action='store_true', help="Store histograms or tree")
parser.add_argument("-c", "--mc", dest="mc", action='store_true', help="MC or data")
parser.add_argument("-r", "--runparallel", dest="runparallel", action='store_true', help="Enable parallel run")
parser.add_argument("-v", "--mva", dest="mva", action='store_true', help="Evaluate MVA")
parser.add_argument("--model", dest="model", default='xgb', help="Type of classifier")
parser.add_argument("--modelfile", dest="modelfile", default='../models/mva.model', help="Name of the classifier file")
parser.add_argument("--phi", action='store_true', help="Run R(phi) analyzer")
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

def analyze(inputfile, outputfile, Analyzer, hist, mc, mva, model, modelfile, parallel=False, outpath='.'):
    if parallel:
        ich, inputfile = inputfile
        print("Processing chunk number %i"%(ich))
        outputfile = outpath+'/'+outputfile.replace('.root','')+'_subset'+str(ich)+'.root'
    analyzer = Analyzer(inputfile, outputfile, hist, mc, mva, model, modelfile)
    analyzer.run()

if __name__ == "__main__":
    from scripts.BToKLLAnalyzer import BToKLLAnalyzer
    from scripts.BToKstarLLAnalyzer import BToKstarLLAnalyzer
    from scripts.BToPhiLLAnalyzer import BToPhiLLAnalyzer
    Analyzer = BToKLLAnalyzer if not args.phi else BToPhiLLAnalyzer

    if '.root' in args.inputfiles:
        fileList = [args.inputfiles,]
    else:
        with open(args.inputfiles) as filenames:
            fileList = [f.rstrip('\n') for f in filenames]

    if not args.runparallel:
        inputfile = fileList
        outputfile = args.outputfile.replace('.root','').replace('.h5','')+'.root'
        analyze(inputfile, outputfile, Analyzer, args.hist, args.mc, args.mva, args.model, args.modelfile)

    else:
        outpath = '.'
        if not os.path.exists(outpath):
            exec_me("mkdir -p %s"%(outpath), False)

        group  = 8
        # stplie files in to n(group) of chunks
        fChunks= list(chunks(fileList,group))
        print ("writing %s jobs"%(len(fChunks)))

        pool = mp.Pool(processes = 4)
        input_parallel = list(enumerate(fChunks))
        partial_func = partial(analyze, outputfile=args.outputfile, Analyzer=Analyzer, hist=args.hist, mc=args.mc, mva=args.mva, model=args.model, modelfile=args.modelfile, parallel=True, outpath=outpath)
        pool.map(partial_func, input_parallel) 

        pool.close()
        pool.join()

        outputfile = args.outputfile.replace('.root','').replace('.h5','')
        exec_me("hadd -k -f %s/%s %s/%s"%(outpath,outputfile+'.root',outpath,outputfile+'_subset*.root'))
        exec_me("rm %s/%s"%(outpath,outputfile+'_subset*.root'))
        

