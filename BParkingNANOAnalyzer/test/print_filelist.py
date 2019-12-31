import subprocess
from os import listdir
from os.path import isfile, join

#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Oct25/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/191025_125913/0000/']
#mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Oct28/BdToKstarJpsi_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarJpsi_Toee/191028_080830/0000/']

mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018A_part2/191021_131326/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018A_part2/191021_131326/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018B_part2/191021_131046/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018B_part2/191021_131046/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018C_part2/191021_131929/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018C_part2/191021_131929/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Nov29/ParkingBPH2/crab_data_Run2018D_part2/191129_152054/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Nov29/ParkingBPH2/crab_data_Run2018D_part2/191129_152054/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Nov29/ParkingBPH2/crab_data_Run2018D_part2/191129_152054/0002/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Nov29/ParkingBPH2/crab_data_Run2018D_part2/191129_152054/0003/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Nov29/ParkingBPH2/crab_data_Run2018D_part2/191129_152054/0004/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Nov29/ParkingBPH2/crab_data_Run2018D_part2/191129_152054/0005/',]

mypath2 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH3/crab_data_Run2018A_part3/191021_131508/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH3/crab_data_Run2018A_part3/191021_131508/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH3/crab_data_Run2018B_part3/191021_131207/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2019Oct21/ParkingBPH3/crab_data_Run2018B_part3/191021_131207/0001/',]

mypath = mypath1 + mypath2 #+ mypath3

#redirector = 'root://eoscms.cern.ch/'
redirector = '/eos/cms/'
#redirector = 'root://cms-xrd-global.cern.ch//'
filelist = []
for path in mypath:
    #filename = subprocess.check_output('xrdfs {} ls {}'.format(redirector, path), shell=True).split('\n')
    #filelist += filename
    #path = '/eos/cms/' + path
    filelist = filelist + [path + f for f in listdir(path) if isfile(join(path, f))]

filelist = ['{}{}'.format(redirector, f.replace('/eos/cms/','')) for f in filelist if ".root" in f]

#outputFile = open('BParkingNANO_2019Sep12_Run2018D_part2.list', 'w+')
outputFile = open('BParkingNANO_2019Oct21_Run2018A2A3B2B3C2D2.list', 'w+')
#outputFile = open('BParkingNANO_2019Oct25_BuToKJpsi_Toee.list', 'w+')
#outputFile = open('BParkingNANO_2019Oct28_BdToKstarJpsi_ToKPiee.list', 'w+')

for f in filelist:
    outputFile.write('%s\n'%(f))
outputFile.close()

