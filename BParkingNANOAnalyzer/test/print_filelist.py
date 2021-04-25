import subprocess
from os import listdir
from os.path import isfile, join
import random

mypath = []

#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BuToKee_MufilterPt6_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee_v2/210307_020324/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BuToKee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee/210307_020206/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee_v2/210307_020844/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BuToKPsi2S_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKPsi2S_Toee_v2/210307_022322/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BdToKstaree_MufilterPt6_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarEE_v2/210307_021235/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BdToKstarJpsi_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarJpsi_Toee_v2/210307_021804/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BdToKstarPsi2S_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarPsi2S_Toee_v2/210307_022818/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/mc_noSkim/BParkingNANO_2021Mar23/BuToKee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee/210323_222613/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/mc_noSkim/BParkingNANO_2021Mar23/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee_v2/210323_222415/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BuToKPsi2S_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKPsi2S_Toee/210307_022211/0000/']
#mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BdToKstarJpsi_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarJpsi_Toee/210307_021655/0000/']
mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar07/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/210307_020712/0000/']

'''
mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018A_part1/210305_221032/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018A_part1/210305_221032/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018B_part1/210305_221634/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018B_part1/210305_221634/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018C_part1/210305_222135/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018C_part1/210305_222135/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018D_part1/210305_222409/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018D_part1/210305_222409/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018D_part1/210305_222409/0002/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018D_part1/210305_222409/0003/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018D_part1/210305_222409/0004/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018D_part1/210305_222409/0005/',
           ]

mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018D_part1_missingLumis/210329_121409/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH1/crab_data_Run2018B_part1_missingLumis/210329_121225/0000/',
          ]

mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018A_part2/210310_142346/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018A_part2/210310_142346/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018B_part2/210310_142257/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018B_part2/210310_142257/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018C_part2/210310_142431/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018C_part2/210310_142431/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018D_part2/210310_145836/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018D_part2/210310_145836/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018D_part2/210310_145836/0002/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018D_part2/210310_145836/0003/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018D_part2/210310_145836/0004/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar10/ParkingBPH2/crab_data_Run2018D_part2/210310_145836/0005/',
           ]

mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018A_part3/210305_222019/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018A_part3/210305_222019/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018B_part3/210305_221932/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018B_part3/210305_221932/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018C_part3/210305_222100/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018C_part3/210305_222100/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018D_part3/210305_222225/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018D_part3/210305_222225/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018D_part3/210305_222225/0002/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018D_part3/210305_222225/0003/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018D_part3/210305_222225/0004/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH3/crab_data_Run2018D_part3/210305_222225/0005/',
           ]

mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018A_part4/210306_185422/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018A_part4/210306_185422/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018B_part4/210306_185522/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018B_part4/210306_185522/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018C_part4/210306_185621/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018C_part4/210306_185621/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018D_part4/210306_185726/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018D_part4/210306_185726/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018D_part4/210306_185726/0002/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018D_part4/210306_185726/0003/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018D_part4/210306_185726/0004/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar06/ParkingBPH4/crab_data_Run2018D_part4/210306_185726/0005/',
           ]

mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018A_part5/210305_215924/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018A_part5/210305_215924/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018B_part5/210305_220057/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018B_part5/210305_220057/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018C_part5/210305_220217/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018C_part5/210305_220217/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018D_part5/210305_220325/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018D_part5/210305_220325/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018D_part5/210305_220325/0002/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018D_part5/210305_220325/0003/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018D_part5/210305_220325/0004/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH5/crab_data_Run2018D_part5/210305_220325/0005/',
           ]

mypath += ['/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH6/crab_data_Run2018A_part6/210305_221453/0000/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH6/crab_data_Run2018A_part6/210305_221453/0001/',
           '/eos/cms/store/group/phys_bphys/bpark/nanoaod_RK2021/BParkingNANO_2021Mar05/ParkingBPH6/crab_data_Run2018B_part6/210305_221938/0000/',
           ]
'''
'''
mypath += ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH1/crab_data_Run2018A_part1/200116_150535/0000/',
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
mypath += ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH2/crab_data_Run2018A_part2/200116_162432/0000/',
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
mypath += ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH3/crab_data_Run2018A_part3/200116_213248/0000/',
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
mypath += ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH4/crab_data_Run2018A_part4/200116_172624/0000/',
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
mypath += ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Feb05/ParkingBPH5/crab_data_Run2018A_part5/200205_165610/0000/',
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
mypath += ['/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH6/crab_data_Run2018A_part6/200116_173128/0000/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH6/crab_data_Run2018A_part6/200116_173128/0001/',
           '/eos/cms/store/group/cmst3/group/bpark/BParkingNANO_2020Jan16/ParkingBPH6/crab_data_Run2018B_part6/200116_173248/0000/',
           ]
'''

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

#outputFile = open('BParkingNANO_2021Mar05_Run2018ABCD.list', 'w+')
outputFile = open('BParkingNANO_2021Mar05_BuToKJpsi_Toee.list', 'w+')
#outputFile = open('BParkingNANO_2021Mar05_BuToKee_trg_noSkim.list', 'w+')
#outputFile = open('BParkingNANO_2020Jan16_Run2018ABCD.list', 'w+')


for f in filelist:
    outputFile.write('%s\n'%(f))
outputFile.close()

