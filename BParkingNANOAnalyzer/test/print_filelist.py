import subprocess
from os import listdir
from os.path import isfile, join

#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018A_part2/190912_154729/0000/',
#           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018A_part2/190912_154729/0001/',]
#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH3/crab_data_Run2018A_part3/190912_154846/0000/',
#           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH3/crab_data_Run2018A_part3/190912_154846/0001/']
#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018B_part2/190912_183627/0000/',
#           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018B_part2/190912_183627/0001/']
#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH3/crab_data_Run2018B_part3/190912_183750/0000/',
#           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH3/crab_data_Run2018B_part3/190912_183750/0001/']
#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018C_part2/190912_155245/0000/',
#           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018C_part2/190912_155245/0001/']
#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH3/crab_data_Run2018C_part3/190912_155407/0000/',
#           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH3/crab_data_Run2018C_part3/190912_155407/0001/']

mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018D_part2/190912_155004/0000/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018D_part2/190912_155004/0001/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018D_part2/190912_155004/0002/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018D_part2/190912_155004/0003/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018D_part2/190912_155004/0004/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018D_part2/190912_155004/0005/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018D_part2/190912_155004/0006/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018D_part2/190912_155004/0007/']

#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/190912_160127/0000/',]

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
outputFile = open('BParkingNANO_2019Sep12_Run2018D_part2_directEOS.list', 'w+')
#outputFile = open('BParkingNANO_2019Sep12_BuToKJpsi_Toee.list', 'w+')

for f in filelist:
    outputFile.write('%s\n'%(f))
outputFile.close()

