#! /usr/bin/env python

import ROOT
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import time
import datetime
import uproot
import awkward
import sys
import os
import numpy as np
import uproot_methods
import pandas as pd

#import rootpy.ROOT as ROOT
from rootpy.io import root_open
from rootpy.plotting import Hist
from rootpy.tree import Tree
from root_numpy import fill_hist, array2root, array2tree
import root_pandas

#ROOT.gErrorIgnoreLevel=ROOT.kError

class BParkingNANOAnalyzer(object):
  def __init__(self, inputfiles, outputfile, inputbranches, outputbranches, hist=False):
    __metaclass__ = ABCMeta
    self._file_out_name = outputfile.replace('.root','').replace('.h5','')
    self._file_in_name = inputfiles
    self._num_files = len(self._file_in_name)
    self._inputbranches = inputbranches
    self._outputbranches = outputbranches
    self._hist = hist
    self._ifile = 0


  def init_output(self):
    print('[BParkingNANOAnalyzer::init_output] INFO: FILE: {}/{}. Initializing output {}...'.format(self._ifile+1, self._num_files, 'histograms' if self._hist else 'tree'))
    # can choose output to be histograms or a tree
    if self._hist:
      self._file_out = root_open(self._file_out_name+'.root', 'recreate')
      self._hist_list = {hist_name: Hist(hist_bins['nbins'], hist_bins['xmin'], hist_bins['xmax'], name=hist_name, title='', type='F') for hist_name, hist_bins in sorted(self._outputbranches.items())}

    else:
      if os.path.isfile(self._file_out_name+'.root'): os.system('rm {}'.format(self._file_out_name+'.root'))


  def fill_output(self):
    print('[BParkingNANOAnalyzer::fill_output] INFO: FILE: {}/{}. Filling the output {}...'.format(self._ifile+1, self._num_files, 'histograms' if self._hist else 'tree'))
    if self._hist:
      for hist_name, hist_bins in sorted(self._outputbranches.items()):
        if hist_name in self._branches.keys():
          branch_np = self._branches[hist_name].values
          fill_hist(self._hist_list[hist_name], branch_np[np.isfinite(branch_np)])
    else:
      self._branches = self._branches[self._outputbranches.keys()]
      #self._branches.to_hdf(self._file_out_name+'.h5', 'branches', mode='a', format='table', append=True)
      self._branches.to_root(self._file_out_name+'.root', key='tree', mode='a')

  def finish(self):
    print('[BParkingNANOAnalyzer::finish] INFO: Merging the output files...')
    #os.system("hadd -k -f {}.root {}_subset*.root".format(self._file_out_name, self._file_out_name))
    #os.system("rm {}_subset*.root".format(self._file_out_name))
    if self._hist:
      for hist_name, hist in sorted(self._hist_list.items()):
        hist.write()
      self._file_out.close()
    else:
      pass


  def print_timestamp(self):
    ts_start = time.time()
    print("[BParkingNANOAnalysis::print_timestamp] INFO : Time: {}".format(datetime.datetime.fromtimestamp(ts_start).strftime('%Y-%m-%d %H:%M:%S')))


  @abstractmethod
  def run(self):
    pass

  def GetDMass(self, row, prefix):
    if (row[prefix+'_k_charge'] * row[prefix+'_l1_charge']) < 0.0:
      return row[prefix+'_Dmass_l1']
    else:
      return row[prefix+'_Dmass_l2']

  def EleEtaCats(self, row, prefix):    
    etaCut = 1.44
    if (abs(row[prefix+'_fit_l1_eta']) < etaCut) and (abs(row[prefix+'_fit_l2_eta']) < etaCut):
      return 0
    elif (abs(row[prefix+'_fit_l1_eta']) > etaCut) and (abs(row[prefix+'_fit_l2_eta']) > etaCut):
      return 1
    else:
      return 2

  def DecayCats(self, row, prefix):    
    mc_matched_selection = (row[prefix+'_l1_genPartIdx'] > -0.5) & (row[prefix+'_l2_genPartIdx'] > -0.5) & (row[prefix+'_k_genPartIdx'] > -0.5)
    # B->K ll
    RK_nonresonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) == 521) & (abs(row[prefix+'_k_genMotherPdgId']) == 521)
    RK_nonresonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_k_genMotherPdgId'] == row[prefix+'_l1_genMotherPdgId'])
    RK_nonresonant_chain_selection &= mc_matched_selection

    # B->K J/psi(ll)
    RK_resonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) == 443) & (abs(row[prefix+'_k_genMotherPdgId']) == 521)
    RK_resonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_k_genMotherPdgId'] == row[prefix+'_l1Mother_genMotherPdgId']) & (row[prefix+'_k_genMotherPdgId'] == row[prefix+'_l2Mother_genMotherPdgId'])
    RK_resonant_chain_selection &= mc_matched_selection

    # B->K*(K pi) ll
    RKstar_nonresonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) == 511) & (abs(row[prefix+'_k_genMotherPdgId']) == 313)
    RKstar_nonresonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_kMother_genMotherPdgId']) 
    RKstar_nonresonant_chain_selection &= mc_matched_selection

    # B->K*(K pi) J/psi(ll)
    RKstar_resonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) == 443) & (abs(row[prefix+'_k_genMotherPdgId']) == 313)
    RKstar_resonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_l1Mother_genMotherPdgId'] == row[prefix+'_kMother_genMotherPdgId']) 
    RKstar_resonant_chain_selection &= mc_matched_selection

    # Bs->phi(K K) ll
    Rphi_nonresonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) == 531) & (abs(row[prefix+'_k_genMotherPdgId']) == 333)
    Rphi_nonresonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_kMother_genMotherPdgId']) 
    Rphi_nonresonant_chain_selection &= mc_matched_selection

    # Bs->phi(K K) J/psi(ll)
    Rphi_resonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) == 443) & (abs(row[prefix+'_k_genMotherPdgId']) == 333)
    Rphi_resonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_l1Mother_genMotherPdgId'] == row[prefix+'_kMother_genMotherPdgId']) 
    Rphi_resonant_chain_selection &= mc_matched_selection

    if RK_nonresonant_chain_selection: return 0
    elif RK_resonant_chain_selection: return 1
    elif RKstar_nonresonant_chain_selection: return 2
    elif RKstar_resonant_chain_selection: return 3
    elif Rphi_nonresonant_chain_selection: return 4
    elif Rphi_resonant_chain_selection: return 5
    else: return -1






