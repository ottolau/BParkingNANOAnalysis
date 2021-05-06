import os,glob
import sys, os, fnmatch
import random
import ROOT

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfiles", dest="inputfiles", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-f", "--suffix", dest="suffix", default=None, help="Suffix of the output name")
parser.add_argument("-c", "--mc", dest="mc", action='store_true', help="MC or data")
parser.add_argument("-m", "--maxevents", dest="maxevents", type=int, default=ROOT.TTree.kMaxEntries, help="Maximum number events to loop over")
parser.add_argument("-v", "--mva", dest="mva", action='store_true', help="Evaluate MVA")
parser.add_argument("--fold", dest="fold", type=int, default=-1, help="Fold number")
parser.add_argument("--model", dest="model", default='xgb', help="Type of classifier")
parser.add_argument("--phi", action='store_true', help="Run R(phi) analyzer")
parser.add_argument("--random", action='store_true', help="Randomize the files' order")
args = parser.parse_args()

def exec_me(command, dryRun=False):
    print(command)
    if not dryRun:
        os.system(command)

def write_condor(exe='runjob.sh', arguments = [], files = [],dryRun=True):
    job_name = exe.replace('.sh','.jdl')
    out = 'universe = vanilla\n'
    out += 'Executable = %s\n'%exe
    out += 'Should_Transfer_Files = YES\n'
    out += 'WhenToTransferOutput = ON_EXIT_OR_EVICT\n'
    out += 'Transfer_Input_Files = %s,%s\n'%(exe,','.join(files))
    out += 'Output = job_%s.stdout\n'%job_name
    out += 'Error  = job_%s.stderr\n'%job_name
    out += 'Log    = job_%s.log\n'   %job_name
    #out += 'request_memory = 8000\n'
    out += 'request_cpus = 4\n'
    out += 'use_x509userproxy = true\n'
    out += 'x509userproxy = $ENV(X509_USER_PROXY)\n' # for lxplus
    out += 'Arguments = %s\n'%(' '.join(arguments))
    #out += '+JobFlavour = "espresso"\n'
    #out += '+JobFlavour = "longlunch"\n'
    #out += '+JobFlavour = "workday"\n'
    out += '+JobFlavour = "tomorrow"\n'
    #out += '+JobFlavour = "testmatch"\n'
    #out += '+MaxRuntime = 36000\n'
    out += 'on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)\n'
    out += 'max_retries = 2\n'
    out += 'requirements = Machine =!= LastRemoteHost\n'
    out += 'Queue 1\n'
    with open(job_name, 'w') as f:
        f.write(out)
    if not dryRun:
        os.system("condor_submit %s"%job_name)

def write_bash(temp = 'runjob.sh', command = '', outputdir = ''):
    out = '#!/bin/bash\n'
    out += 'date\n'
    out += 'MAINDIR=`pwd`\n'
    out += 'export HOME=${MAINDIR}\n'
    out += 'export EOS_MGM_URL=root://eosuser.cern.ch\n'
    out += 'ls\n'
    out += '#CMSSW from scratch (only need for root)\n'
    out += 'export CWD=${PWD}\n'
    out += 'export PATH=${PATH}:/cvmfs/cms.cern.ch/common\n'
    out += 'export CMS_PATH=/cvmfs/cms.cern.ch\n'
    out += 'export SCRAM_ARCH=slc7_amd64_gcc700\n'
    out += 'source /cvmfs/cms.cern.ch/cmsset_default.sh\n'
    out += 'scramv1 project CMSSW CMSSW_10_2_15\n'
    out += 'cd CMSSW_10_2_15/src\n'
    out += 'eval `scramv1 runtime -sh` # cmsenv\n'
    out += 'virtualenv myenv\n'
    out += 'source myenv/bin/activate\n'
    #out += 'eval `scramv1 runtime -sh` # cmsenv\n'
    #out += 'git clone git@github.com:ottolau/BParkingNANOAnalysis.git\n'
    out += 'mv ${MAINDIR}/BParkingNANOAnalysis.tgz .\n'
    out += 'tar -xf BParkingNANOAnalysis.tgz\n'
    out += 'rm BParkingNANOAnalysis.tgz\n'
    out += 'cd BParkingNANOAnalysis\n'
    out += 'cp ${MAINDIR}/setup_condor.sh BParkingNANOAnalyzer/setup_condor.sh\n'
    out += '. BParkingNANOAnalyzer/setup_condor.sh\n'
    #out += 'eval `scramv1 runtime -sh` # cmsenv\n'
    out += 'scram b clean; scram b\n'
    out += 'cd BParkingNANOAnalyzer/test\n'
    out += command + '\n'
    #if args.hist:
    #  out += 'echo "List all root files = "\n'
    #  out += 'ls *.root\n'
    out += 'echo "List all files"\n'
    out += 'ls\n'
    out += 'echo "*******************************************"\n'
    out += 'OUTDIR='+outputdir+'\n'
    out += 'echo "xrdcp output for condor"\n'
    out += 'for FILE in *.{}\n'.format('root')
    out += 'do\n'
    out += '  echo "xrdcp -f ${FILE} ${OUTDIR}/${FILE}"\n'
    out += '  xrdcp -f ${FILE} ${OUTDIR}/${FILE} 2>&1\n'
    out += '  XRDEXIT=$?\n'
    out += '  if [[ $XRDEXIT -ne 0 ]]; then\n'
    out += '    rm *.{}\n'.format('root')
    out += '    echo "exit code $XRDEXIT, failure in xrdcp"\n'
    out += '    exit $XRDEXIT\n'
    out += '  fi\n'
    out += '  rm ${FILE}\n'
    out += 'done\n'
    out += 'cd $MAINDIR\n'
    out += 'echo "Inside $MAINDIR:"\n'
    out += 'ls\n'
    out += 'echo "DELETING..."\n'
    out += 'rm -rf CMSSW_10_2_15\n'
    out += 'ls\n'
    out += 'date\n'
    with open(temp, 'w') as f:
        f.write(out)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':
    basePath = "."
    sampleFolders = os.listdir(basePath)
    inputfiles = args.inputfiles.replace('.list','')
    inputfiles += '_BToPhiEEAnalyzer' if args.phi else '_BToKEEAnalyzer'
    if args.suffix is not None: inputfiles += '_{}'.format(args.suffix)
    if args.mc: inputfiles += '_mc'
    if args.mva: inputfiles += '_mva'
    if args.random: inputfiles += '_random'
    if args.fold != -1: inputfiles += '_fold{}'.format(args.fold)

    outputBase = "output_{}".format(inputfiles)
    outputDir = 'root://eosuser.cern.ch//eos/user/k/klau/BParkingNANO_forCondor/output/{}'.format(inputfiles)
    outputName = inputfiles

    dryRun  = False
    subdir  = os.path.expandvars("$PWD")
    group   = 600
    #group   = 50
    #group = 30

    zipPath = 'zip'
    if not os.path.exists(zipPath):
      exec_me("mkdir -p {}".format(zipPath), False)
    exec_me("git clone https://github.com/ottolau/BParkingNANOAnalysis.git {}".format(os.path.join(zipPath, "BParkingNANOAnalysis")), False)
    exec_me("tar -zcvf BParkingNANOAnalysis.tgz -C {} {}".format(zipPath, "BParkingNANOAnalysis"), False)

    files = ['../../setup_condor.sh', '../../scripts/helper.py', '../../scripts/BToKLLAnalyzer.py', '../../scripts/BToPhiLLAnalyzer.py', '../runAnalyzer.py', 'BParkingNANOAnalysis.tgz']
    if args.mva:
        files += ['../../models/mva.model']
    files_condor = [f.split('/')[-1] for f in files]

    fileList = []
    with open(args.inputfiles) as filenames:
        fileList = [f.rstrip('\n') for f in filenames]

    if args.random:
        print('Shuffling the input files...')
        #random.shuffle(fileList)
        fileList = random.sample(fileList, k=500)
        #fileList = random.sample(fileList, k=8000)

    # stplie files in to n(group) of chunks
    fChunks= list(chunks(fileList,group))
    print("writing %s jobs"%(len(fChunks)))

    #for i in range(group):
    for i, fChunk in enumerate(fChunks):
        outpath  = "%s/sub_%d/"%(outputBase,i)
        if not os.path.exists(outpath):
            exec_me("mkdir -p %s"%(outpath), False)
        os.chdir(os.path.join(subdir,outpath))
        print(os.getcwd())
        for f in files:
            exec_me("cp %s/%s ."%(subdir,f), False)

        inputfileList = open('inputfile_%d.list'%(i), 'w+')

        for f in fChunk:
            inputfileList.write('%s\n'%(f))
        inputfileList.close()

        args_list = []
        if args.random: args_list.append('--random')
        if args.mc: args_list.append('-c')
        if args.mva: 
            args_list.append('-v')
            args_list.append('--model {}'.format(args.model))
        if args.phi: args_list.append('--phi')
        if args.fold != -1:
            args_list.append('-f {}'.format(args.fold))
        args_str = " ".join(args_list)

        cmd = ""
        cmd += "cp ${{MAINDIR}}/inputfile_{0}.list .;".format(i)
        cmd += "cp ${MAINDIR}/BToKLLAnalyzer.py ${MAINDIR}/CMSSW_10_2_15/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/scripts/;"
        cmd += "cp ${MAINDIR}/BToPhiLLAnalyzer.py ${MAINDIR}/CMSSW_10_2_15/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/scripts/;"
        cmd += "cp ${MAINDIR}/helper.py ${MAINDIR}/CMSSW_10_2_15/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/scripts/;"
        if args.mva:
            cmd += "cp ${MAINDIR}/mva.model ${MAINDIR}/CMSSW_10_2_15/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/models/;"
        cmd += "cp ${MAINDIR}/runAnalyzer.py ${MAINDIR}/CMSSW_10_2_15/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/test/;"
        cmd += "python runAnalyzer.py -i inputfile_{0}.list -o {1}_subset{0}.root -r {2}".format(i,outputName,args_str if len(args_list) > 0 else "")

        inputargs =  []
        f_sh = "runjob_%s.sh"%i
        cwd    = os.getcwd()
        write_bash(f_sh, cmd, outputDir)
        write_condor(f_sh ,inputargs, files_condor + ['inputfile_%d.list'%(i)], dryRun)
        os.chdir(subdir)

    exec_me("rm -rf {}".format(os.path.join(zipPath, "BParkingNANOAnalysis")), False)
    exec_me("rm -rf BParkingNANOAnalysis.tgz", False)

