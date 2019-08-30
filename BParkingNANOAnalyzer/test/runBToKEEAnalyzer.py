import ROOT
import os
import multiprocessing as mp
import sys
#print(sys.path)
sys.path.append('../')
#print(sys.path)
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
    for i in range(0, len(l), n):
        yield l[i:i + n]

def analyze(tchain, outputfile, hist):
    analyzer = BToKLLAnalyzer(tchain, outputfile)
    analyzer.loop(-1, hist)

def analyzeParallel(enumfChunk):
    ich, fChunk = enumfChunk
    print("Processing chunk number %i"%(ich))
    tchain = ROOT.TChain(args.ttree)
    for filename in fChunk:
        tchain.Add(filename)

    outputfile = outpath+'/'+args.outputfile.replace('.root','')+'_subset'+str(ich)+'.root'
    analyze(tchain, outputfile, args.hist)


if __name__ == "__main__":
    if not args.runparallel:
        tchain = ROOT.TChain(args.ttree)
        with open(args.inputfiles) as filenames:
            for filename in filenames:
                tchain.Add(filename.rstrip('\n'))
                #break
        #tchain.Add('testBParkNANO_data_10215.root')
        outputfile = args.outputfile.replace('.root','')+'.root'
        analyze(tchain, outputfile, args.hist)

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
        outputfile = args.outputfile.replace('.root','')
        exec_me("hadd -k -f %s/%s %s/%s"%(outpath,outputfile+'.root',outpath,outputfile+'_subset*.root'))
        exec_me("rm %s/%s"%(outpath,outputfile+'_subset*.root'))


