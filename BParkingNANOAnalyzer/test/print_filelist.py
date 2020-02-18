import subprocess
from os import listdir
from os.path import isfile, join

#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/200116_215618/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/BuToKee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee/200116_215859/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/BdToKstaree_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarEE/200116_220807/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/BdToKstarJpsi_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarJpsi_Toee/200116_220459/0000/']


mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018A_part1/200116_150535/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018A_part1/200116_150535/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018B_part1/200116_150810/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018B_part1/200116_150810/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018C_part1/200116_151112/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018C_part1/200116_151112/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018D_part1/200116_151214/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018D_part1/200116_151214/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018D_part1/200116_151214/0002/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018D_part1/200116_151214/0003/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018D_part1/200116_151214/0004/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018D_part1/200116_151214/0005/',
           ]

mypath2 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018A_part2/200116_162432/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018A_part2/200116_162432/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018B_part2/200116_162718/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018B_part2/200116_162718/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018C_part2/200116_162917/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018C_part2/200116_162917/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018D_part2/200116_165722/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018D_part2/200116_165722/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018D_part2/200116_165722/0002/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018D_part2/200116_165722/0003/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018D_part2/200116_165722/0004/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018D_part2/200116_165722/0005/',
           ]

mypath3 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH3/crab_data_Run2018A_part3/200116_213248/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH3/crab_data_Run2018A_part3/200116_213248/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH3/crab_data_Run2018B_part3/200116_213159/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH3/crab_data_Run2018B_part3/200116_213159/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH3/crab_data_Run2018C_part3/200116_213339/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH3/crab_data_Run2018C_part3/200116_213339/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan17/ParkingBPH3/crab_data_Run2018D_part3/200117_074527/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan17/ParkingBPH3/crab_data_Run2018D_part3/200117_074527/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan17/ParkingBPH3/crab_data_Run2018D_part3/200117_074527/0002/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan17/ParkingBPH3/crab_data_Run2018D_part3/200117_074527/0003/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan17/ParkingBPH3/crab_data_Run2018D_part3/200117_074527/0004/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan17/ParkingBPH3/crab_data_Run2018D_part3/200117_074527/0005/',
           ]

mypath4 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018A_part4/200116_172624/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018A_part4/200116_172624/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018B_part4/200116_172741/0000/',
           #'/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018B_part4/200116_172741/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018C_part4/200116_172852/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018C_part4/200116_172852/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018D_part4/200116_173014/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018D_part4/200116_173014/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018D_part4/200116_173014/0002/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018D_part4/200116_173014/0003/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018D_part4/200116_173014/0004/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018D_part4/200116_173014/0005/',
           ]

mypath5 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb05/ParkingBPH5/crab_data_Run2018A_part5/200205_165610/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb05/ParkingBPH5/crab_data_Run2018A_part5/200205_165610/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb05/ParkingBPH5/crab_data_Run2018B_part5/200205_165723/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb05/ParkingBPH5/crab_data_Run2018B_part5/200205_165723/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb05/ParkingBPH5/crab_data_Run2018C_part5/200205_170223/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb05/ParkingBPH5/crab_data_Run2018C_part5/200205_170223/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb07/ParkingBPH5/crab_data_Run2018D_part5/200207_191502/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb07/ParkingBPH5/crab_data_Run2018D_part5/200207_191502/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb07/ParkingBPH5/crab_data_Run2018D_part5/200207_191502/0002/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb07/ParkingBPH5/crab_data_Run2018D_part5/200207_191502/0003/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb07/ParkingBPH5/crab_data_Run2018D_part5/200207_191502/0004/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb07/ParkingBPH5/crab_data_Run2018D_part5/200207_191502/0005/',
           ]

mypath6 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH6/crab_data_Run2018A_part6/200116_173128/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH6/crab_data_Run2018A_part6/200116_173128/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH6/crab_data_Run2018B_part6/200116_173248/0000/',
           ]

#mypath1 = ['/eos/user/k/klau/BParkingNANO_forCondor/output/BParkingNANO_2020Jan16_Run2018ABCDpartial_BToKEEAnalyzer_2020Feb15/']

mypath = mypath1 + mypath2 + mypath3 + mypath4 + mypath5 + mypath6

#redirector = 'root://eoscms.cern.ch/'
#redirector = '/eos/cms/'
redirector = 'root://cms-xrd-global.cern.ch//'
#redirector = ''
filelist = []
for path in mypath:
    #filename = subprocess.check_output('xrdfs {} ls {}'.format(redirector, path), shell=True).split('\n')
    #filelist += filename
    #path = '/eos/cms/' + path
    filelist = filelist + [path + f for f in listdir(path) if isfile(join(path, f))]

filelist = ['{}{}'.format(redirector, f.replace('/eos/cms/','')) for f in filelist if ".root" in f]

#outputFile = open('BParkingNANO_2019Sep12_Run2018D_part2.list', 'w+')
#outputFile = open('BParkingNANO_2019Oct21_Run2018A2A3B2B3C2D2.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKJpsi_Toee.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKee.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BdToKstarJpsi_ToKPiee.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BdToKstaree.list', 'w+')
outputFile = open('BParkingNANO_2020Jan16_Run2018ABCD.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_Run2018ABCDpartial_BToKEEAnalyzer_2020Feb15.list', 'w+')

for f in filelist:
    outputFile.write('%s\n'%(f))
outputFile.close()

