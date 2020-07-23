import subprocess
from os import listdir
from os.path import isfile, join
import random

#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/200116_215618/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/BuToKee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee/200116_215859/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/BdToKstaree_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarEE/200116_220807/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/BdToKstarJpsi_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarJpsi_Toee/200116_220459/0000/']
#mypath2 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb26/BuToKee_MufilterPt6_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee_ext/200226_210729/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb26/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/200226_153915/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Mar22/BuToKPsi2S_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKPsi2S_Toee/200322_185610/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Mar22/BdToKstarPsi2S_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarPsi2S_Toee/200322_185719/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Mar31/BuToKee_MufilterPt6_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee_ext/200331_183927/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Mar31/ParkingBPH2/crab_data_Run2018B_part2/200331_183424/0000/',
#           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Mar31/ParkingBPH2/crab_data_Run2018B_part2/200331_183424/0001/',]
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr11/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/200411_170141/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr11/BuToKee_MufilterPt6_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee_ext/200411_170027/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr13/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/200413_014042/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr13/BuToKee_MufilterPt6_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee_ext/200413_013817/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr07/BuToKee_MufilterPt6_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee_ext/200407_210002/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr16/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/200416_042500/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr19/BuToKee_MufilterPt6_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee_ext/200419_021400/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr19/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/200419_021637/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr19/ParkingBPH4/crab_data_Run2018B_part4/200419_194523/0000/',
#           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr19/ParkingBPH4/crab_data_Run2018B_part4/200419_194523/0001/',
#           ]
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr19/BsToPhiee_MufilterPt3_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BsToPhiEE/200419_193103/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr19/BsToPhiJpsi_ToKKee_MufilterPt2_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BsToPhiJpsi_Toee/200419_193225/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr30/BsToPhiee_MufilterPt3_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BsToPhiEE/200429_221635/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Apr30/BsToPhiJpsi_ToKKee_MufilterPt2_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BsToPhiJpsi_Toee/200429_221930/0000/']

#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/BsToPhiee_MufilterPt3_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BsToPhiEE/200522_034641/0000/']
#mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/BsToPhiJpsi_ToKKee_MufilterPt2_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BsToPhiJpsi_Toee/200522_034755/0000/']
mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jul02/ParkingBPH1/crab_data_Run2018B_part1/200701_221618/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jul02/ParkingBPH1/crab_data_Run2018B_part1/200701_221618/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jun30/ParkingBPH2/crab_data_Run2018B_part2/200630_203700/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jun30/ParkingBPH2/crab_data_Run2018B_part2/200630_203700/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jun30/ParkingBPH3/crab_data_Run2018B_part3/200630_203815/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jun30/ParkingBPH3/crab_data_Run2018B_part3/200630_203815/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jul02/ParkingBPH4/crab_data_Run2018B_part4/200701_221732/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jul02/ParkingBPH4/crab_data_Run2018B_part4/200701_221732/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jul02/ParkingBPH5/crab_data_Run2018B_part5/200701_221854/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jul02/ParkingBPH5/crab_data_Run2018B_part5/200701_221854/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jul02/ParkingBPH6/crab_data_Run2018B_part6/200701_221949/0000/',
           ]
'''
mypath1 = ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018A_part2/200522_193404/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018A_part2/200522_193404/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018B_part2/200522_035112/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018B_part2/200522_035112/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018C_part2/200522_193552/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018C_part2/200522_193552/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018D_part2/200522_193656/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018D_part2/200522_193656/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018D_part2/200522_193656/0002/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018D_part2/200522_193656/0003/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018D_part2/200522_193656/0004/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020May22/ParkingBPH2/crab_data_Run2018D_part2/200522_193656/0005/',
           ]
'''
'''
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
'''

mypath = mypath1 #+ mypath2 + mypath3 + mypath4 + mypath5 + mypath6

#redirector = 'root://eoscms.cern.ch/'
#redirector = '/eos/cms/'
redirector = 'root://cms-xrd-global.cern.ch//'
#redirector = ''
filelist = []
for path in mypath:
    #filename = subprocess.check_output('xrdfs {} ls {}'.format(redirector, path), shell=True).split('\n')
    #filelist += filename
    filename = subprocess.check_output('eos ls {}'.format(path), shell=True).split('\n')
    filelist += [path + f for f in filename]
    #path = '/eos/cms/' + path
    #filelist = filelist + [path + f for f in listdir(path) if isfile(join(path, f))]

#filelist = random.sample(filelist, k=30)
filelist = ['{}{}'.format(redirector, f.replace('/eos/cms/','')) for f in filelist if ".root" in f]

#outputFile = open('BParkingNANO_2019Sep12_Run2018D_part2.list', 'w+')
#outputFile = open('BParkingNANO_2019Oct21_Run2018A2A3B2B3C2D2.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKJpsi_Toee.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKee.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BdToKstarJpsi_ToKPiee.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BdToKstaree.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_Run2018ABCD_random30.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKee_all.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKJpsi_Toee_svprob0.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKPsi2S_Toee.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BdToKstarPsi2S_ToKPiee.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKee_ext_yutaPR.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_Run2018B2_yutaPR.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKJpsi_Toee_rmUnbiased.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_BuToKee_ext_rmUnbiased.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_Run2018B4_rmUnbiased.list', 'w+')
#outputFile = open('BParkingNANO_2020May22_BsToPhiee.list', 'w+')
#outputFile = open('BParkingNANO_2020May22_BsToPhiJpsi_Toee.list', 'w+')
#outputFile = open('BParkingNANO_2020May22_Run2018ABCD2_BsToPhiLL.list', 'w+')
outputFile = open('BParkingNANO_2020Jan16_Run2018B_likesign.list', 'w+')

for f in filelist:
    outputFile.write('%s\n'%(f))
outputFile.close()

