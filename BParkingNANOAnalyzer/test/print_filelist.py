import subprocess

'''
mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep08/ParkingBPH2/crab_data_Run2018A_part2/190908_173415/0000/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep08/ParkingBPH2/crab_data_Run2018A_part2/190908_173415/0001/']

mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep10/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/190910_155408/0000/']
mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep10/ParkingBPH2/crab_data_Run2018B_part2/190910_082846/0000/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep10/ParkingBPH2/crab_data_Run2018B_part2/190910_082846/0001/',]
'''

mypath1 = ['/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018A_part2/190912_154729/0000/',
           '/store/group/cmst3/group/bpark/BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018A_part2/190912_154729/0000/',]

mypath = mypath1 #+ mypath2 #+ mypath3

redirector = 'root://eoscms.cern.ch/'
filelist = []
for path in mypath:
    filename = subprocess.check_output('xrdfs {} ls {}'.format(redirector, path), shell=True).split('\n')
    filelist += filename

filelist = ['{}{}'.format(redirector, f) for f in filelist if ".root" in f]

outputFile = open('BParkingNANO_2019Sep12_Run2018A_part2.list', 'w+')
#outputFile = open('BParkingNANO_2019Sep08_BuToKJpsi_Toee.list', 'w+')
#outputFile = open('BParkingNANO_2019Sep10_Run2018B_part2.list', 'w+')

for f in filelist:
    outputFile.write('%s\n'%(f))
outputFile.close()

