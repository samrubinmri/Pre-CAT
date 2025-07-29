## Example CEST scenario configs (2-pool and 3-pool examples)
# Adapted from code by Nikita Vladimirov 
# https://github.com/momentum-laboratory/molecular-mrf

import numpy as np

# Define CEST parameters here:
# ################
# Scanner parameters
# ################
B0 = 7 # T
inhomB0 = 0 # B0 inhomogeneity
relB1 = 1 # Relative B1

# ################
# Water pool (a)
# ################
T1 = np.arange(1300.0, 2600.0 + 100, 100) / 1000  # (s)
T2 = np.arange(40.0, 130.0 + 10, 10) / 1000  # (s)
waterProtonConcentration = 111e3 # mM
# ################
# Solute pool (b)
# ################
pool_name = 'Cr'
wb_ppm = 2.0  # Chemical shift of the CEST pool in [ppm]
T1b = np.nan  # (s) 
T2b = 1e-3  # (s)
numOfExchangeableProtonsSolute = 4.0  # Number of exchangeable solute protons
soluteConcentration = np.arange(2.0, 100.0 + 2.0, 2.0)  # Solute concentration (mM)
fb = soluteConcentration * numOfExchangeableProtonsSolute / waterProtonConcentration  # Proton fraction [0,1]
kb = np.arange(100.0, 500.0 + 5.0, 5.0)  # Solute chemical exchange rate (s^-1)
# ################
# MT pool (c)
# ################
wc_ppm = -2.5  # Chemical shift of the MT pool in [ppm]
T1c = np.nan  # (s) # Temporary set here as nan, will later be equal to and vary with water T1
T2c = 40.0 / 1e6  # (s) # Note that this is altered for Lorentzian and not super-Loretnzian
numOfExchangeableProtonsMT = 1.0  # Number of exchangeable solute protons
MTConcentration = np.arange(2e3, 30e3 + 2e3, 2e3)  # Solute concentration (mM)
fc = MTConcentration * numOfExchangeableProtonsMT / waterProtonConcentration  # Proton fraction [0,1]
kc = np.arange(5.0, 100 + 5.0, 5.0) # Solute chemical exchange rate (s^-1)

# ################
# Additional parameters
# ################
scale = 1
resetInitMag = 0
verbose = 0
maxPulseSamples = 100
numWorkers = 18

class Config:
	def get_config(self):
		return self.cfg 

class ConfigExample(Config):
	def __init__(self):
		config = {}
		config['yaml_fn'] = 'example.yaml'
		config['seq_fn'] = 'example.seq'
		config['dict_fn'] = 'example.mat'

		# Water pool
		config['water_pool'] = {}
		config['water_pool']['t1'] = T1.tolist()
		config['water_pool']['t2'] = T2.tolist()
		config['water_pool']['f'] = 1

		# Solute pool 
		config['cest_pool'][pool_name] = {}
		config['cest_pool'][pool_name]['t1'] = 
		