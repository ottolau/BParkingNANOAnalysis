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
parser.add_argument("-v", "--mva", dest="mva", action='store_true', help="Evaluate MVA")
parser.add_argument("-r", "--runparallel", dest="runparallel", action='store_true', help="Enable parallel run")
parser.add_argument("--phi", action='store_true', help="Run R(phi) analyzer")
parser.add_argument("--saveSeparate", dest="saveSeparate", action='store_true', help="Save separate files")
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

def analyze(inputfile, outputfile, Analyzer, mva, parallel=False, outpath='.'):
    if parallel:
        ich, inputfile = inputfile
        print("Processing chunk number %i"%(ich))
        outputfile = outpath+'/'+outputfile.replace('.root','')+'_subset'+str(ich)+'.root'
    analyzer = Analyzer(inputfile, outputfile, mva)
    analyzer.run()


if __name__ == "__main__":
    from scripts.BToKLLAnalyzer_postprocess import BToKLLAnalyzer_postprocess
    from scripts.BToPhiLLAnalyzer_postprocess import BToPhiLLAnalyzer_postprocess
    Analyzer = BToKLLAnalyzer_postprocess if not args.phi else BToPhiLLAnalyzer_postprocess

    if '.root' in args.inputfiles:
        fileList = [args.inputfiles,]
    else:
        with open(args.inputfiles) as filenames:
            fileList = [f.rstrip('\n') for f in filenames]

    if not args.runparallel:
        inputfile = fileList
        outputfile = args.outputfile.replace('.root','').replace('.h5','')+'.root'
        analyze(inputfile, outputfile, Analyzer, args.mva)

    else:
        global outpath
        outpath = '.'
        outputfolder = 'RootFiles'
        if not os.path.exists(outpath):
            exec_me("mkdir -p %s"%(outpath), False)

        group  = 8
        # stplie files in to n(group) of chunks
        fChunks= list(chunks(fileList,group))
        print ("writing %s jobs"%(len(fChunks)))

        pool = mp.Pool(processes = 8)
        input_parallel = list(enumerate(fChunks))
        partial_func = partial(analyze, outputfile=args.outputfile, Analyzer=Analyzer, mva=args.mva, parallel=True, outpath=outpath)
        pool.map(partial_func, input_parallel) 

        outputfile = args.outputfile.replace('.root','').replace('.h5','')
        exec_me("hadd -k -f %s/%s %s/%s"%(outpath,outputfile+'.root',outpath,outputfile+'_subset*.root'))

        if args.saveSeparate:
          path = os.path.join(outpath, outputfolder, outputfile)
          exec_me("mkdir -p {}".format(path))
          exec_me("mv {} {}".format(outputfile+'_subset*.root', path))

          redirector = ''
          filelist = [path + '/' + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

          outputFile = open(outpath+'/'+outputfile+'.list', 'w+')

          for f in filelist:
              outputFile.write('%s\n'%(f))
          outputFile.close()

        else:
          exec_me("rm %s/%s"%(outpath,outputfile+'_subset*.root'))


