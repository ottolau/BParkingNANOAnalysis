import subprocess
from os import listdir
from os.path import isfile, join

#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Oct25/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/191025_125913/0000/']
mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Oct28/BdToKstarJpsi_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarJpsi_Toee/191028_080830/0000/']

mypath = mypath1 #+ mypath2 #+ mypath3

#redirector = 'root://eoscms.cern.ch/'
#redirector = '/eos/cms/'
redirector = 'root://cms-xrd-global.cern.ch/'
filelist = []
for path in mypath:
    #filename = subprocess.check_output('xrdfs {} ls {}'.format(redirector, path), shell=True).split('\n')
    #filelist += filename
    path = '/eos/cms/' + path
    filelist = filelist + [path + f for f in listdir(path) if isfile(join(path, f))]

filelist = ['{}{}'.format(redirector, f).replace('/eos/cms/','') for f in filelist if ".root" in f]

#outputFile = open('BParkingNANO_2019Sep12_Run2018D_part2.list', 'w+')
#outputFile = open('BParkingNANO_2019Sep12_Run2018A2A3B2B3C2C3D2.list', 'w+')
#outputFile = open('BParkingNANO_2019Oct25_BuToKJpsi_Toee.list', 'w+')
outputFile = open('BParkingNANO_2019Oct28_BdToKstarJpsi_ToKPiee.list', 'w+')

for f in filelist:
    outputFile.write('%s\n'%(f))
outputFile.close()

