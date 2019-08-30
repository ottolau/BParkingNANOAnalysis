import ROOT
import os
import multiprocessing as mp
import sys

from BParkingNANOAnalysis.BParkingNANOAnalyzer.BToKLLAnalyzer import BToKLLAnalyzer


import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfiles", dest="inputfiles", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-m", "--maxevents", dest="maxevents", type=int, default=ROOT.TTree.kMaxEntries, help="Maximum number events to loop over")
parser.add_argument("-t", "--ttree", dest="ttree", default="Events", help="TTree Name")
parser.add_argument("-r", "--runparallel", dest="runparallel", default=False, help="Enable PROOF")
args = parser.parse_args()


ROOT.gROOT.SetBatch()

def exec_me(command, dryRun=False):
    print(command)
    if not dryRun:
        os.system(command)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def analyze(tchain, outputfile):
    analyzer = BToKLLAnalyzer(tchain, outputfile)
    analyzer.loop(-1)

def analyzeParallel(enumfChunk):
    ich, fChunk = enumfChunk
    print("Processing chunk number %i"%(ich))
    tchain = ROOT.TChain(args.ttree)
    for filename in fChunk:
        tchain.Add(filename)
    print('Total number of events: ' + str(tchain.GetEntries()))

    outputfile = outpath+'/'+args.outputfile+'_subset'+str(ich)+'.root'
    analyze(tchain, outputfile)


if __name__ == "__main__":
    if not args.runparallel:
        tchain = ROOT.TChain(args.ttree)
        with open(args.inputfiles) as filenames:
            for filename in filenames:
                tchain.Add(filename.rstrip('\n'))
                #break
        #tchain.Add('testBParkNANO_data_10215.root')
        outputfile = args.outputfile
        analyze(tchain, outputfile)

    else:
        outputBase = "/eos/uscms/store/user/klau/BsPhiLL_output/LowPtElectronSculpting"
        outputFolder = "BsPhiEE_CutBasedEvaluation"
        global outputpath
        #outpath  = "%s/%s"%(outputBase,outputFolder)
        outpath = '.'
        if not os.path.exists(outpath):
            exec_me("mkdir -p %s"%(outpath), False)

        with open(args.inputfiles) as filenames:
            fileList = [f.rstrip('\n') for f in filenames]
        group   = 15
        # stplie files in to n(group) of chunks
        fChunks= list(chunks(fileList,group))
        print ("writing %s jobs for %s"%(len(fChunks),outputFolder))
        pool = mp.Pool(processes = 8)
        pool.map(analyzeParallel, enumerate(fChunks))
        exec_me("hadd -k %s/%s %s/%s"%(outpath,args.outputfile+'.root',outpath,args.outputfile+'_subset*.root'))
        exec_me("rm %s/%s"%(outpath,args.outputfile+'_subset*.root'))


