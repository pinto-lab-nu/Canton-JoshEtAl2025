# ==============================
# ======  Dependencies  ========
# ==============================
#
# PintoLab_dj 
# PintoLab_imagingAnalysis
# PintoLab_utils
# these can all be installed as packages


# ==============================
# ======  Import stuff  ========
# ==============================
from utils import connect_to_dj
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
VM = connect_to_dj.get_virtual_modules()


# ==============================
# ======= Local modules ========
# ==============================
code_dir = "/Users/lpr6177/Documents/code/Canton-JoshEtAl2025/"
sys.path.insert(0,code_dir)
from analyzeSpont2P import params as tau_params
import analyzeSpont2P

# ==============================
# ==========  Fig 1  ===========
# ==============================


# ==============================
# ==========  Fig 2  ===========
# ==============================

fig2 = plt.plot()

# fig 2a: FOV and cranial window egs
plt.subplot(231)

# fig 2b: auto corr + exp egs
plt.subplot(232)

# fig 2c: 2p tau summary
plt.subplot(233)
ax = plt.gca()
v1_taus, v1_keys, v1_total = analyzeSpont2P.get_all_tau('V1', params = tau_params, dff_type = 'residuals_dff')
m2_taus, m2_keys, m2_total = analyzeSpont2P.get_all_tau('M2', params = tau_params, dff_type = 'residuals_dff')
tau_stats, ax_tau = analyzeSpont2P.plot_area_tau_comp(v1_taus=v1_taus, m2_taus=m2_taus, axisHandle = ax, params = tau_params)
tau_stats['V1_total_num_cells'] = v1_total
tau_stats['M2_total_num_cells'] = m2_total