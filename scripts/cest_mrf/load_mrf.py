#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:42:34 2025

@author: jonah
"""
import numpy as np
import scripts.BrukerMRI as bruker
import pypulseq

def seq_from_method(num, directory, cfg):
	"""
	Create Pypulseq object from method and config file parameters.
	"""
	data = bruker.ReadExperiment(directory, num)
	method = data.method 
	seq_defs = {}
	seq_defs['n_pulses'] = method['PVM_MagTransPulseNumb']  # Number of pulses
	seq_defs['tp'] = method['Fp_SatDur'] * 1e-3  # Pulse duration [s]
	seq_defs['td'] = method['PVM_MagTransInterDelay'] * 1e-3  # Interpulse delay [s]
	seq_defs['Trec'] = method['Fp_TRDels']  # recovery time [s]
	seq_defs['Trec_M0'] = 'NaN'  # Recovery time with respect to m0 if exist [s]
	seq_defs['M0_offset'] = 'NaN'  # Dummy M0 offset [ppm]
	seq_defs['DCsat'] = seq_defs['tp'] / (seq_defs['tp'] + seq_defs['td'])  # Duty cycle
	seq_defs['offsets_ppm'] = np.round(method['Fp_SatOffset'] / method['PVM_FrqWork'][0], 2) # Offset vector [ppm]
	seq_defs['num_meas'] = len(seq_defs['offsets_ppm'])  # number of repetition
	seq_defs['Tsat'] = seq_defs['n_pulses'] * (seq_defs['tp'] + seq_defs['td']) - seq_defs['td']
	seq_defs['B1pa'] = method['Fp_SatPows']
	seq_defs['B0'] = cfg['b0']  # B0 [T]
