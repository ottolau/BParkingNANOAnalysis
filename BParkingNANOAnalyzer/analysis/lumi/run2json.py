import json
import numpy as np
import os

###
# Get a txt file with run number
# 'dasgoclient --query='run dataset=/ParkingBPH2/Run2018D-05May2019promptD-v1/MINIAOD' | tee -a BPark_Run2018A2A3B2B3C2C3D2_run.txt'
###

run_file = 'BPark_Run2018D2_run.txt'
run_output = 'BPark_Run2018D2_runjson.txt'
cert_json = 'Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt'
output_json = 'BPark_Run2018D2_JSON.txt'

run = np.loadtxt(run_file, dtype='int')
run_dict = {"{}".format(str(run_num)): [[1,99999],] for run_num in run}
print(run_dict)
if os.path.isfile(run_output): os.system('rm {}'.format(run_output))

with open(run_output, 'w') as json_file:
    json.dump(run_dict, json_file)

os.system('compareJSON.py --and {} {} {}'.format(run_output, cert_json, output_json))
    
