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

  def get_events(self, filename, checkbranch='nBToKEE'):
    events = uproot.open(filename)['Events']
    return events if checkbranch in events.allkeys() else None

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
      self._branches = self._branches[self._outputbranches.keys()].sort_index(axis=1)
      self._branches.to_root(self._file_out_name+'.root', key='tree', mode='a', store_index=False)

  def finish(self):
    print('[BParkingNANOAnalyzer::finish] INFO: Merging the output files...')
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

  def EleEtaCats(self, l1_eta, l2_eta):
    etaCut = 1.44
    if (abs(l1_eta) < etaCut) and (abs(l2_eta) < etaCut):
      return 0
    elif (abs(l1_eta) > etaCut) and (abs(l2_eta) > etaCut):
      return 1
    else:
      return 2

  def DecayCats(self, row, prefix):    
    mc_matched_selection = (row[prefix+'_l1_genPartIdx'] > -0.5) & (row[prefix+'_l2_genPartIdx'] > -0.5) & (row[prefix+'_k_genPartIdx'] > -0.5)
    # B->K ll
    RK_nonresonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) == 521) & (abs(row[prefix+'_k_genMotherPdgId']) == 521)
    RK_nonresonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_k_genMotherPdgId'] == row[prefix+'_l1_genMotherPdgId'])
    RK_nonresonant_chain_selection &= mc_matched_selection

    # B->K J/psi(ll) or B->K psi(2S)(ll)
    RK_resonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) in {443, 100443}) & (abs(row[prefix+'_k_genMotherPdgId']) == 521)
    RK_resonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_k_genMotherPdgId'] == row[prefix+'_l1Mother_genMotherPdgId']) & (row[prefix+'_k_genMotherPdgId'] == row[prefix+'_l2Mother_genMotherPdgId'])
    RK_resonant_chain_selection &= mc_matched_selection

    # B->K*(K pi) ll
    RKstar_nonresonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) == 511) & (abs(row[prefix+'_k_genMotherPdgId']) == 313)
    RKstar_nonresonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_kMother_genMotherPdgId']) 
    RKstar_nonresonant_chain_selection &= mc_matched_selection

    # B->K*(K pi) J/psi(ll) or B->K*(K pi) psi(2S)(ll)
    RKstar_resonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) in {443, 100443}) & (abs(row[prefix+'_k_genMotherPdgId']) == 313)
    RKstar_resonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_l1Mother_genMotherPdgId'] == row[prefix+'_kMother_genMotherPdgId']) 
    RKstar_resonant_chain_selection &= mc_matched_selection

    # Bs->phi(K K) ll
    Rphi_nonresonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) == 531) & (abs(row[prefix+'_k_genMotherPdgId']) == 333)
    Rphi_nonresonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_kMother_genMotherPdgId']) 
    Rphi_nonresonant_chain_selection &= mc_matched_selection

    # Bs->phi(K K) J/psi(ll) or Bs->phi(K K) psi(2S)(ll)
    Rphi_resonant_chain_selection = (abs(row[prefix+'_l1_genMotherPdgId']) in {443, 100443}) & (abs(row[prefix+'_k_genMotherPdgId']) == 333)
    Rphi_resonant_chain_selection &= (row[prefix+'_l1_genMotherPdgId'] == row[prefix+'_l2_genMotherPdgId']) & (row[prefix+'_l1Mother_genMotherPdgId'] == row[prefix+'_kMother_genMotherPdgId']) 
    Rphi_resonant_chain_selection &= mc_matched_selection

    if RK_nonresonant_chain_selection: return 0
    elif RK_resonant_chain_selection: return 1
    elif RKstar_nonresonant_chain_selection: return 2
    elif RKstar_resonant_chain_selection: return 3
    elif Rphi_nonresonant_chain_selection: return 4
    elif Rphi_resonant_chain_selection: return 5
    else: return -1

  def DecayCats_vectorized(self, l1_genPartIdx, l2_genPartIdx, k_genPartIdx,
                l1_genPdgId, l2_genPdgId, k_genPdgId,
                l1_genMotherPdgId, l2_genMotherPdgId, k_genMotherPdgId,
                l1Mother_genMotherPdgId, l2Mother_genMotherPdgId, kMother_genMotherPdgId):    

    mc_matched_selection = (l1_genPartIdx > -0.5) & (l2_genPartIdx > -0.5) & (k_genPartIdx > -0.5) & (abs(l1_genPdgId) in {11, 13}) & (abs(l2_genPdgId) in {11, 13})
    # B->K ll
    RK_nonresonant_chain_selection = (abs(k_genPdgId) == 321) & (abs(l1_genMotherPdgId) == 521) & (abs(k_genMotherPdgId) == 521)
    RK_nonresonant_chain_selection &= (l1_genMotherPdgId == l2_genMotherPdgId) & (k_genMotherPdgId == l1_genMotherPdgId)
    RK_nonresonant_chain_selection &= mc_matched_selection

    # B->K J/psi(ll) or B->K psi(2S)(ll)
    RK_resonant_chain_selection = (abs(k_genPdgId) == 321) & (abs(l1_genMotherPdgId) in {443, 100443}) & (abs(k_genMotherPdgId) == 521)
    RK_resonant_chain_selection &= (l1_genMotherPdgId == l2_genMotherPdgId) & (k_genMotherPdgId == l1Mother_genMotherPdgId) & (k_genMotherPdgId == l2Mother_genMotherPdgId)
    RK_resonant_chain_selection &= mc_matched_selection

    # B->K*(K pi) ll
    RKstar_nonresonant_chain_selection = (abs(l1_genMotherPdgId) == 511) & (abs(k_genMotherPdgId) == 313)
    RKstar_nonresonant_chain_selection &= (l1_genMotherPdgId == l2_genMotherPdgId) & (l1_genMotherPdgId == kMother_genMotherPdgId) 
    RKstar_nonresonant_chain_selection &= mc_matched_selection

    # B->K*(K pi) J/psi(ll) or B->K*(K pi) psi(2S)(ll)
    RKstar_resonant_chain_selection = (abs(l1_genMotherPdgId) in {443, 100443}) & (abs(k_genMotherPdgId) == 313)
    RKstar_resonant_chain_selection &= (l1_genMotherPdgId == l2_genMotherPdgId) & (l1Mother_genMotherPdgId == kMother_genMotherPdgId) 
    RKstar_resonant_chain_selection &= mc_matched_selection

    # Bs->phi(K K) ll
    Rphi_nonresonant_chain_selection = (abs(k_genPdgId) == 321) & (abs(l1_genMotherPdgId) == 531) & (abs(k_genMotherPdgId) == 333)
    Rphi_nonresonant_chain_selection &= (l1_genMotherPdgId == l2_genMotherPdgId) & (l1_genMotherPdgId == kMother_genMotherPdgId) 
    Rphi_nonresonant_chain_selection &= mc_matched_selection

    # Bs->phi(K K) J/psi(ll) or Bs->phi(K K) psi(2S)(ll)
    Rphi_resonant_chain_selection = (abs(k_genPdgId) == 321) & (abs(l1_genMotherPdgId) in {443, 100443}) & (abs(k_genMotherPdgId) == 333)
    Rphi_resonant_chain_selection &= (l1_genMotherPdgId == l2_genMotherPdgId) & (l1Mother_genMotherPdgId == kMother_genMotherPdgId) 
    Rphi_resonant_chain_selection &= mc_matched_selection

    if RK_nonresonant_chain_selection: return 0
    elif RK_resonant_chain_selection: return 1
    elif RKstar_nonresonant_chain_selection: return 2
    elif RKstar_resonant_chain_selection: return 3
    elif Rphi_nonresonant_chain_selection: return 4
    elif Rphi_resonant_chain_selection: return 5
    else: return -1

  def DecayCats_BToPhiLL_vectorized(self, l1_genPartIdx, l2_genPartIdx, trk1_genPartIdx, trk2_genPartIdx,
                                    l1_genPdgId, l2_genPdgId, trk1_genPdgId, trk2_genPdgId,
                                    l1_genMotherPdgId, l2_genMotherPdgId, trk1_genMotherPdgId, trk2_genMotherPdgId,
                                    l1Mother_genMotherPdgId, l2Mother_genMotherPdgId, trk1Mother_genMotherPdgId, trk2Mother_genMotherPdgId):    

    mc_matched_selection = (l1_genPartIdx > -0.5) & (l2_genPartIdx > -0.5) & (trk1_genPartIdx > -0.5) & (trk2_genPartIdx > -0.5) & (abs(l1_genPdgId) in {11, 13}) & (abs(l2_genPdgId) in {11, 13})

    # Bs->phi(K K) ll
    Rphi_nonresonant_chain_selection = (abs(trk1_genPdgId) == 321) & (abs(trk2_genPdgId) == 321) & (abs(l1_genMotherPdgId) == 531) & (abs(trk1_genMotherPdgId) == 333) & (abs(trk2_genMotherPdgId) == 333)
    Rphi_nonresonant_chain_selection &= (l1_genMotherPdgId == l2_genMotherPdgId) & (trk1_genMotherPdgId == trk2_genMotherPdgId) & (l1_genMotherPdgId == trk1Mother_genMotherPdgId) 
    Rphi_nonresonant_chain_selection &= mc_matched_selection

    # Bs->phi(K K) J/psi(ll) or Bs->phi(K K) psi(2S)(ll)
    Rphi_resonant_chain_selection = (abs(trk1_genPdgId) == 321) & (abs(trk2_genPdgId) == 321) & (abs(l1_genMotherPdgId) in {443, 100443}) & (abs(trk1_genMotherPdgId) == 333) & (abs(trk2_genMotherPdgId) == 333)
    Rphi_resonant_chain_selection &= (l1_genMotherPdgId == l2_genMotherPdgId) & (trk1_genMotherPdgId == trk2_genMotherPdgId) & (l1Mother_genMotherPdgId == trk1Mother_genMotherPdgId) 
    Rphi_resonant_chain_selection &= mc_matched_selection

    if Rphi_nonresonant_chain_selection: return 0
    elif Rphi_resonant_chain_selection: return 1
    else: return -1

  def DeltaPhi(self, phi1, phi2):
    dphi = phi1 - phi2
    if dphi >= np.pi: dphi -= 2.0*np.pi
    elif dphi < -1.0*np.pi: dphi += 2.0*np.pi
    return dphi



