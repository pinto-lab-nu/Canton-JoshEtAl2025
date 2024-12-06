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

# fig 2b: auto corr + exp egs (or maybe this goes in fig 1)
plt.subplot(232)

# fig 2c: 2p tau summary
plt.subplot(233)
ax = plt.gca()
v1_taus, v1_keys, v1_total = analyzeSpont2P.get_all_tau('V1', params = tau_params, dff_type = 'residuals_dff')
m2_taus, m2_keys, m2_total = analyzeSpont2P.get_all_tau('M2', params = tau_params, dff_type = 'residuals_dff')
tau_stats, ax_tau = analyzeSpont2P.plot_area_tau_comp(v1_taus=v1_taus, m2_taus=m2_taus, axisHandle = ax, params = tau_params)
tau_stats['V1_total_num_cells'] = v1_total
tau_stats['M2_total_num_cells'] = m2_total

# fig 2d, e: x-corr

# fig 2f: clustering FOV example
v1_centr, v1_rec_ids = analyzeSpont2P.get_centroids_by_rec(v1_keys)
m2_centr, m2_rec_ids = analyzeSpont2P.get_centroids_by_rec(m2_keys)
fov_v1 = analyzeSpont2P.plot_tau_fov(v1_keys, v1_rec_ids, which_sess=2, do_zscore=False, prctile_cap=[0,95])
fov_m2 = analyzeSpont2P.plot_tau_fov(v1_keys, v1_rec_ids, which_sess=17, do_zscore=False, prctile_cap=[0,95])
# good ones m2: 0, 4, 5, 10 (95th prct), 17, 22
# good ones v1: 0 , 2 

# fig 2g: clustering
clust_stats_v1 , tau_diff_mat_v1 = analyzeSpont2P.clustering_by_tau(v1_taus, v1_centr, v1_rec_ids, params = tau_params)
clust_stats_m2 , tau_diff_mat_m2 = analyzeSpont2P.clustering_by_tau(m2_taus, m2_centr, m2_rec_ids, params = tau_params)
cax = analyzeSpont2P.plot_clustering_comp(v1_clust=clust_stats_v1,m2_clust=clust_stats_m2, params = tau_params)
# %%


# try just long or short timescale cells for clusetring