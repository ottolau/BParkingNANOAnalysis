import os,glob
import sys, commands, os, fnmatch
from optparse import OptionParser

def exec_me(command, dryRun=False):
    print command
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
    out += 'use_x509userproxy = true\n'
    out += 'Arguments = %s\n'%(' '.join(arguments))
    out += 'Queue 1\n'
    with open(job_name, 'w') as f:
        f.write(out)
    if not dryRun:
        os.system("condor_submit %s"%job_name)

def write_bash(temp = 'runjob.sh', command = ''):
    out = '#!/bin/bash\n'
    out += 'date\n'
    out += 'MAINDIR=`pwd`\n'
    out += 'ls\n'
    out += '#CMSSW from scratch (only need for root)\n'
    out += 'export CWD=${PWD}\n'
    out += 'export PATH=${PATH}:/cvmfs/cms.cern.ch/common\n'
    out += 'export CMS_PATH=/cvmfs/cms.cern.ch\n'
    out += 'export SCRAM_ARCH=slc6_amd64_gcc700\n'
    out += 'scramv1 project CMSSW CMSSW_10_2_15\n'
    out += 'cd CMSSW_10_2_15/src\n'
    out += 'eval `scramv1 runtime -sh` # cmsenv\n'
    out += 'git clone git://github.com/ottolau/BParkingNANOAnalysis.git\n'
    out += 'cd BParkingNANOAnalysis\n'
    out += 'scram b clean; scram b\n'
    out += 'cd BParkingNANOAnalyzer/test\n'
    out += command + '\n'
    out += 'echo "List all root files = "\n'
    out += 'ls *.root\n'
    out += 'echo "List all files"\n'
    out += 'ls\n'
    out += 'echo "*******************************************"\n'
    out += 'OUTDIR=root://cmseos.fnal.gov//store/user/klau/BParkingNANO_forCondor/output\n'
    out += 'echo "xrdcp output for condor"\n'
    out += 'for FILE in *.root\n'
    out += 'do\n'
    out += '  echo "xrdcp -f ${FILE} ${OUTDIR}/${FILE}"\n'
    out += '  xrdcp -f ${FILE} ${OUTDIR}/${FILE} 2>&1\n'
    out += '  XRDEXIT=$?\n'
    out += '  if [[ $XRDEXIT -ne 0 ]]; then\n'
    out += '    rm *.root\n'
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
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

if __name__ == '__main__':
    basePath = "."
    sampleFolders = os.listdir(basePath)    
    outputBase = "test"
    dryRun  = False
    subdir  = os.path.expandvars("$PWD")
    group   = 1
    files = ["test.list"]
   
    for i in range(group):
        outpath  = "%s/sub_%d/"%(outputBase,i)
        if not os.path.exists(outpath):
            exec_me("mkdir -p %s"%(outpath), False)
        os.chdir(os.path.join(subdir,outpath))
        print  os.getcwd()
        for f in files:
            exec_me("cp %s/%s ."%(subdir,f), False)
        cmd = "cp ${MAINDIR}/test.list .; python runBToKEEAnalyzer.py -i test.list" 

        args =  []
        f_sh = "runjob_%s.sh"%i
        cwd    = os.getcwd()
        write_bash(f_sh, cmd)
        write_condor(f_sh ,args, files, dryRun)
        os.chdir(subdir)

