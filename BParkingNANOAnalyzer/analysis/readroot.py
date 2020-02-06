import pandas as pd
import numpy as np
from rootpy.io import root_open
from rootpy.plotting import Hist, Hist2D
from root_numpy import fill_hist, array2root
from root_pandas import to_root
import uproot
import time
import xgboost as xgb
import gc

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfile", dest="inputfile", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-s", "--hist", dest="hist", action='store_true', help="Store histograms or tree")
args = parser.parse_args()


outputbranches = {'BToKEE_mll_raw': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.5},
                  'BToKEE_mll_fullfit': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.5},
                  'BToKEE_mll_llfit': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.5},
                  'BToKEE_q2': {'nbins': 100, 'xmin': 0.0, 'xmax': 25.0},
                  'BToKEE_fit_mass': {'nbins': 100, 'xmin': 4.5, 'xmax': 6.0},
                  'BToKEE_fit_massErr': {'nbins': 100, 'xmin': 0.0, 'xmax': 0.5},
                  'BToKEE_fit_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_fit_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_b_iso03_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_b_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_fit_l1_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_l2_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_l1_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_l2_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_l1_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_fit_l2_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_fit_l1_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_fit_l2_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_l1_dxy_sig': {'nbins': 100, 'xmin': -30.0, 'xmax': 30.0},
                  'BToKEE_l2_dxy_sig': {'nbins': 100, 'xmin': -30.0, 'xmax': 30.0},
                  'BToKEE_l1_dz': {'nbins': 100, 'xmin': -1.0, 'xmax': 1.0},
                  'BToKEE_l2_dz': {'nbins': 100, 'xmin': -1.0, 'xmax': 1.0},
                  #'BToKEE_l1_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  #'BToKEE_l2_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  #'BToKEE_l1_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  #'BToKEE_l2_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l1_mvaId': {'nbins': 100, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l2_mvaId': {'nbins': 100, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l1_pfmvaId': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                  'BToKEE_l2_pfmvaId': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                  'BToKEE_l1_pfmvaCats': {'nbins': 2, 'xmin': 0.0, 'xmax': 2.0},
                  'BToKEE_l2_pfmvaCats': {'nbins': 2, 'xmin': 0.0, 'xmax': 2.0},
                  'BToKEE_l1_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                  'BToKEE_l2_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                  'BToKEE_l1_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                  'BToKEE_l2_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                  'BToKEE_l1_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l2_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l1_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l2_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l1_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l2_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l1_iso03_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_l2_iso03_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_l1_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_l2_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_fit_k_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_fit_k_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_fit_k_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_fit_k_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_k_DCASig': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_k_dz': {'nbins': 100, 'xmin': -1.0, 'xmax': 1.0},
                  'BToKEE_k_nValidHits': {'nbins': 100, 'xmin': 0.0, 'xmax': 100.0},
                  'BToKEE_k_iso03_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_k_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_svprob': {'nbins': 100, 'xmin': 0.0, 'xmax': 1.0},
                  'BToKEE_fit_cos2D': {'nbins': 100, 'xmin': 0.999, 'xmax': 1.0},
                  'BToKEE_l_xy_sig': {'nbins': 100, 'xmin': 0.0, 'xmax': 50.0},
                  'BToKEE_Dmass': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                  'BToKEE_pill_mass': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                  'BToKEE_maxDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                  'BToKEE_minDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                  'BToKEE_eleEtaCats': {'nbins': 3, 'xmin': 0.0, 'xmax': 3.0},
                  'BToKEE_event': {'nbins': 10, 'xmin': 0.0, 'xmax': 10.0},
                  #'BToKEE_fit_mass_decorr': {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0},
                  #'BToKEE_mll_fullfit_decorr': {'nbins': 50, 'xmin': -0.2, 'xmax': 0.2},
                  }

outputbranches_mva = {'BToKEE_xgb': {'nbins': 50, 'xmin': -20.0, 'xmax': 20.0},
                     }

outputhist2d = {'BToKEE_fit_mass_vs_BToKEE_q2': {'nbinx': 50, 'xmin': 4.5, 'xmax': 6.0, 'nbiny': 50, 'ymin': 0.0, 'ymax': 20.0},
                'BToKEE_fit_mass_vs_BToKEE_mll_fullfit': {'nbinx': 50, 'xmin': 4.5, 'xmax': 6.0, 'nbiny': 50, 'ymin': 0.0, 'ymax': 4.5},
                #'BToKEE_fit_mass_decorr_vs_BToKEE_mll_fullfit_decorr': {'nbinx': 50, 'xmin': -1.0, 'xmax': 1.0, 'nbiny': 50, 'ymin': -0.2, 'ymax': 0.2},
               }

ELECTRON_MASS = 0.000511
K_MASS = 0.493677
JPSI_MC = 3.08844
JPSI_SIGMA_MC = 0.04571
JPSI_LOW = np.sqrt(6.0)
JPSI_UP = JPSI_MC + 3.0*JPSI_SIGMA_MC
JPSI_DECORR_MC = 2.5001e-03
JPSI_DECORR_SIGMA_MC = 1.4728e-02
JPSI_DECORR_LOW = JPSI_DECORR_MC - 3.0*JPSI_DECORR_SIGMA_MC
JPSI_DECORR_UP = JPSI_DECORR_MC + 3.0*JPSI_DECORR_SIGMA_MC
B_MC = 5.2694
B_SIGMA_MC = 0.0591
B_UP = B_MC + 3.0*B_SIGMA_MC
B_MIN = 4.5
B_MAX = 6.0
D_MASS_CUT = 1.9

def EleEtaCats(row):    
    etaCut = 1.44
    if (abs(row['BToKEE_fit_l1_eta']) < etaCut) and (abs(row['BToKEE_fit_l2_eta']) < etaCut):
      return 0
    elif (abs(row['BToKEE_fit_l1_eta']) > etaCut) and (abs(row['BToKEE_fit_l2_eta']) > etaCut):
      return 1
    else:
      return 2

if __name__ == "__main__":
  inputfile = args.inputfile.replace('.root','').replace('.h5','')+'.root'
  outputfile = args.outputfile.replace('.root','').replace('.h5','')

  ele_type = {'all': False, 'pf': True, 'low': False, 'mix': False, 'low_pfveto': False, 'mix_net': False}
  ele_selection = {'all': 'all_selection', 'pf': 'pf_selection', 'low': 'low_selection', 'mix': 'mix_selection', 'low_pfveto': 'low_pfveto_selection', 'mix_net': 'mix_net_selection'}

  isMVAEvaluate = True
  prepareMVA = False
  plot2D = True
  isGetDecorr = False

  if isMVAEvaluate:
    #features = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi', 'BToKEE_k_DCASig', 'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig']

    features = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz',
                'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 
                'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi', 'BToKEE_k_DCASig', 'BToKEE_k_dz',
                'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig',
                ]

    features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
    #features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']
    features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']

    training_branches = sorted(features)

    outputbranches.update(outputbranches_mva)
    #mvaCut = 5.14
    mvaCut = 0.0
    ntree_limit = 800
    model = xgb.Booster({'nthread': 6})
    model.load_model('xgb_fulldata_05Feb2020_Dveto_fullq2_EB_pf_isoPFMVA.model')

  if ".list" in args.inputfile:
    with open(args.inputfile) as filenames:
        fileList = [f.rstrip('\n') for f in filenames]
  else:
    fileList = [args.inputfile,]

  for (ifile, filename) in enumerate(fileList):
    print('INFO: FILE: {}/{}. Loading file...'.format(ifile+1, len(fileList)))
    events = uproot.open(filename)['tree']
    #events = uproot.open(inputfile)['tree']

    #params = events.arrays()
    startTime = time.time()
    #for i, params in enumerate(events.iterate(entrysteps=1000000)):
    for i, branches in enumerate(events.iterate(outputtype=pd.DataFrame, entrysteps=1000000)):
      print('Reading chunk {}... Finished opening file in {} s'.format(i, time.time() - startTime))
      #branches = pd.DataFrame(params).sort_index(axis=1)
      #branches.sort_index(axis=1, inplace=True)
      branches['BToKEE_l1_pfmvaId_lowPt'] = np.where(branches['BToKEE_l1_pfmvaCats'] == 0, branches['BToKEE_l1_pfmvaId'], 20.0)
      branches['BToKEE_l2_pfmvaId_lowPt'] = np.where(branches['BToKEE_l2_pfmvaCats'] == 0, branches['BToKEE_l2_pfmvaId'], 20.0)
      branches['BToKEE_l1_pfmvaId_highPt'] = np.where(branches['BToKEE_l1_pfmvaCats'] == 1, branches['BToKEE_l1_pfmvaId'], 20.0)
      branches['BToKEE_l2_pfmvaId_highPt'] = np.where(branches['BToKEE_l2_pfmvaCats'] == 1, branches['BToKEE_l2_pfmvaId'], 20.0)

      '''
      mll_mean = np.mean(branches['BToKEE_mll_fullfit']) if isGetDecorr else 3.00233006477
      fit_mass_mean = np.mean(branches['BToKEE_fit_mass']) if isGetDecorr else 5.17608833313
      branches['BToKEE_mll_fullfit_centered'] = branches['BToKEE_mll_fullfit'] - mll_mean
      branches['BToKEE_fit_mass_centered'] = branches['BToKEE_fit_mass'] - fit_mass_mean
      data_centered = np.array([branches['BToKEE_fit_mass_centered'],branches['BToKEE_mll_fullfit_centered']]).T
      if isGetDecorr:
        CovMatrix = np.cov(data_centered, rowvar=False, bias=True)
        eigVals, eigVecs = np.linalg.eig(CovMatrix)
      else:
        eigVecs = np.array([[0.74743269, -0.66433754], [0.66433754, 0.74743269]])

      print('Rotation matrix: {}'.format(eigVecs))
      #print('BToKEE_mll_fullfit mean: {}, BToKEE_fit_mass mean: {}'.format(np.mean(branches['BToKEE_mll_fullfit']), np.mean(branches['BToKEE_fit_mass'])))
      data_decorr = data_centered.dot(eigVecs)
      branches['BToKEE_fit_mass_decorr'] = data_decorr[:,0]
      branches['BToKEE_mll_fullfit_decorr'] = data_decorr[:,1]
      '''

      jpsi_selection = (branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (branches['BToKEE_mll_fullfit'] < JPSI_UP)
      #jpsi_selection = (branches['BToKEE_mll_fullfit'] > np.sqrt(1.1)) & (branches['BToKEE_mll_fullfit'] < np.sqrt(6.0)) #JPSI_UP)
      #jpsi_selection = (branches['BToKEE_mll_fullfit'] > np.sqrt(1.1)) & (branches['BToKEE_mll_fullfit'] < JPSI_UP)
      #jpsi_selection = (branches['BToKEE_mll_fullfit_decorr'] > JPSI_DECORR_LOW) & (branches['BToKEE_mll_fullfit_decorr'] < JPSI_DECORR_UP)
      #b_selection = jpsi_selection & (branches['BToKEE_fit_mass'] > B_LOWSB_UP) & (branches['BToKEE_fit_mass'] < B_UPSB_LOW)
      b_upsb_selection = (branches['BToKEE_fit_mass'] > B_UP)
      d_veto_selection = branches['BToKEE_Dmass'] > D_MASS_CUT

      #sv_selection = (branches['BToKEE_pt'] > 10.0) & (branches['BToKEE_l_xy_sig'] > 6.0 ) & (branches['BToKEE_svprob'] > 0.1) & (branches['BToKEE_cos2D'] > 0.999)
      l1_selection = (branches['BToKEE_l1_mvaId'] > 4.24) #& (np.logical_not(branches['BToKEE_l1_isPFoverlap']))
      l2_selection = (branches['BToKEE_l2_mvaId'] > 4.24) #& (np.logical_not(branches['BToKEE_l2_isPFoverlap']))
      #k_selection = (branches['BToKEE_k_pt'] > 1.0) #& (branches['BToKEE_k_DCASig'] > 2.0)
      #additional_selection = (branches['BToKEE_mass'] > B_LOW) & (branches['BToKEE_mass'] < B_UP)
      #general_selection = jpsi_selection & sv_selection & k_selection & (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94)

      #general_selection = (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94)
      #general_selection = (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94) & jpsi_selection & d_veto_selection
      #general_selection = (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94) & jpsi_selection & d_veto_selection & (branches['BToKEE_eleEtaCats'] == 2)
      general_selection = d_veto_selection
      general_selection &= (branches['BToKEE_eleEtaCats'] == 0)
      general_selection &= l1_selection & l2_selection
      general_selection &= jpsi_selection
      #general_selection &= (branches['BToKEE_fit_mass'] > 5.0)
      #general_selection &= (branches['BToKEE_l1_pfmvaCats'] == 1) & (branches['BToKEE_l2_pfmvaCats'] == 1)
      #general_selection &= b_upsb_selection
      #general_selection &= np.logical_not(branches['BToKEE_k_isKaon'])

      branches = branches[general_selection]

      # additional cuts, allows various lengths

      l1_pf_selection = (branches['BToKEE_l1_isPF'])
      l2_pf_selection = (branches['BToKEE_l2_isPF'])
      l1_low_selection = (branches['BToKEE_l1_isLowPt']) #& (branches['BToKEE_l1_pt'] < 5.0)
      l2_low_selection = (branches['BToKEE_l2_isLowPt']) #& (branches['BToKEE_l2_pt'] < 5.0)

      pf_selection = l1_pf_selection & l2_pf_selection # & (branches['BToKEE_k_pt'] > 1.5) & (branches['BToKEE_pt'] > 10.0)
      low_selection = l1_low_selection & l2_low_selection
      overlap_veto_selection = np.logical_not(branches['BToKEE_l1_isPFoverlap']) & np.logical_not(branches['BToKEE_l2_isPFoverlap'])
      mix_selection = ((l1_pf_selection & l2_low_selection) | (l2_pf_selection & l1_low_selection))
      low_pfveto_selection = low_selection & overlap_veto_selection
      mix_net_selection = overlap_veto_selection & np.logical_not(pf_selection | low_selection)
      all_selection = pf_selection | low_pfveto_selection | mix_net_selection 

      # count the number of b candidates passes the selection
      #count_selection = jpsi_selection 
      #nBToKEE_selected = branches['BToKEE_event'][count_selection].values
      #_, nBToKEE_selected = np.unique(nBToKEE_selected[np.isfinite(nBToKEE_selected)], return_counts=True)

      if isMVAEvaluate:
        #branches = branches[pf_selection]
        branches['BToKEE_xgb'] = model.predict(xgb.DMatrix(branches[training_branches].sort_index(axis=1).values), ntree_limit=ntree_limit)
        branches = branches[(branches['BToKEE_xgb'] > mvaCut)].sort_values('BToKEE_xgb', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')

      output_branches = {}

      for eType, eBool in ele_type.items():
        if not eBool: continue
        if eType == 'all':
          output_branches[eType] = branches.copy()
        else:
          output_branches[eType] = branches[eval(ele_selection[eType])]

        if args.hist:
          file_out = root_open('{}_{}.root'.format(outputfile, eType), 'recreate')
          hist_list = {hist_name: Hist(hist_bins['nbins'], hist_bins['xmin'], hist_bins['xmax'], name=hist_name, title='', type='F') for hist_name, hist_bins in sorted(outputbranches.items())}
          for hist_name, hist_bins in sorted(outputbranches.items()):
            if hist_name in branches.keys():
              branch_np = output_branches[eType][hist_name].values
              fill_hist(hist_list[hist_name], branch_np[np.isfinite(branch_np)])
              hist_list[hist_name].write()
          if plot2D:
            hist2d_list = {hist_name: Hist2D(hist_bins['nbinx'], hist_bins['xmin'], hist_bins['xmax'], hist_bins['nbiny'], hist_bins['ymin'], hist_bins['ymax'], name=hist_name, title='', type='F') for hist_name, hist_bins in sorted(outputhist2d.items())}
            for hist_name in hist2d_list.keys():
              xvar, yvar = hist_name.split('_vs_')
              xvar_np = output_branches[eType][xvar].values
              yvar_np = output_branches[eType][yvar].values
              fill_hist(hist2d_list[hist_name], np.vstack((xvar_np[np.isfinite(xvar_np)], yvar_np[np.isfinite(yvar_np)])).T)
              hist2d_list[hist_name].write()
          file_out.close()

        else:
          if prepareMVA:
            output_branches[eType] = output_branches[eType].sample(frac=1)
            frac = 0.75
            training_branches = output_branches[eType].iloc[:int(frac*output_branches[eType].shape[0])]
            testing_branches = output_branches[eType].iloc[int(frac*output_branches[eType].shape[0]):]
            training_branches[outputbranches.keys()].to_root('{}_training_{}.root'.format(outputfile, eType), key='tree', mode='a')
            testing_branches[outputbranches.keys()].to_root('{}_testing_{}.root'.format(outputfile, eType), key='tree', mode='a')

          else:
            output_branches[eType][outputbranches.keys()].to_root('{}_{}.root'.format(outputfile, eType), key='tree', mode='a')

      startTime = time.time()
      gc.collect()



