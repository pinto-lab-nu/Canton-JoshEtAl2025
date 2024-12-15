# =============================
# =========  Set up ===========
# =============================

# ======  Dependencies  ========
# PintoLab_dj 
# PintoLab_imagingAnalysis
# PintoLab_utils
# these can all be installed as packages

# ======  Import stuff  ========
from utils import connect_to_dj
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
VM = connect_to_dj.get_virtual_modules()

# ======= Local modules ========
code_dir = "/Users/lpr6177/Documents/code/Canton-JoshEtAl2025/"
sys.path.insert(0,code_dir)
from analyzeSpont2P import params as tau_params
import analyzeSpont2P
from analyzeEvoked2P import params as opto_params
import analyzeEvoked2P

# %% =========================================
# === Fig 1: widefield + merfish analysis ====
# ============================================

# fig 1a widefield schematics

# fig 1b Cux2 histology (or maybe suppl)

# fig 1c raw data with running plus exponential egs

# fig 1d widefield tau map

# fig 1e tau area summary

# fig 1f regression schematics

# fig 1g regression results 

# === Fig S1: tau regression simulations ===

# === Fig S2: merfish regression controls ===

# %% =========================================
# ===== Fig 2: spontaneous 2p timescales =====
# ============================================

fig2 = plt.plot()

# fig 2a: FOV and cranial window egs with raw traces
plt.subplot(231)

# fig 2b: 2p tau summary
plt.subplot(232)

ax = plt.gca()
v1_taus, v1_keys, v1_total = analyzeSpont2P.get_all_tau('V1', params = tau_params, dff_type = 'residuals_dff')
m2_taus, m2_keys, m2_total = analyzeSpont2P.get_all_tau('M2', params = tau_params, dff_type = 'residuals_dff')
tau_stats, ax_tau = analyzeSpont2P.plot_area_tau_comp(v1_taus=v1_taus, m2_taus=m2_taus, axis_handle = ax, params = tau_params)
tau_stats['V1_total_num_cells'] = v1_total
tau_stats['M2_total_num_cells'] = m2_total

# fig 2c, d: x-corr?
plt.subplot(233)


# fig 2e: clustering FOV example
v1_centr, v1_rec_ids = analyzeSpont2P.get_centroids_by_rec(v1_keys)
m2_centr, m2_rec_ids = analyzeSpont2P.get_centroids_by_rec(m2_keys)
fov_v1 = analyzeSpont2P.plot_tau_fov(v1_keys, v1_rec_ids, which_sess=2, do_zscore=False, prctile_cap=[0,95])
fov_m2 = analyzeSpont2P.plot_tau_fov(v1_keys, v1_rec_ids, which_sess=17, do_zscore=False, prctile_cap=[0,95])
# good ones m2: 0, 4, 5, 10 (95th prct), 17, 22
# good ones v1: 0 , 2 

# fig 2f: clustering
clust_stats_v1 , tau_diff_mat_v1 = analyzeSpont2P.clustering_by_tau(v1_taus, v1_centr, v1_rec_ids, params = tau_params)
clust_stats_m2 , tau_diff_mat_m2 = analyzeSpont2P.clustering_by_tau(m2_taus, m2_centr, m2_rec_ids, params = tau_params)
cax = analyzeSpont2P.plot_clustering_comp(v1_clust=clust_stats_v1,m2_clust=clust_stats_m2, params = tau_params)

# >>>>>> try just long or short timescale cells for clusetring

# %% ===========================================
# == Fig 3: overall evoked 2p-opto comparison ==
# ==============================================

# fig 3a raw evoked traces and/or pixel-wise dff example (or suppl movie)

# fig 3b area eg response map

# fig 3c overall response probability
resp_prob_summary, _ = analyzeEvoked2P.plot_response_stats_comparison(params=opto_params, 
                                                                      expt_type='standard', 
                                                                      resp_type='dff', 
                                                                      axis_handle=None, 
                                                                      plot_what='response_probability')

# fig 3d response magnitude
full_resp_stats, _ = analyzeEvoked2P.plot_response_stats_comparison(params=opto_params, 
                                                                    expt_type='standard', 
                                                                    resp_type='dff', 
                                                                    which_neurons='non_stimd', 
                                                                    response_stats=None, 
                                                                    axis_handle=None, 
                                                                    plot_what='response_magnitude')

# fig 3e response vs distance
_, _ = analyzeEvoked2P.plot_response_stats_comparison(params=opto_params, 
                                                      expt_type='standard', 
                                                      resp_type='dff', 
                                                      which_neurons='non_stimd', 
                                                      response_stats=full_resp_stats, 
                                                      axis_handle=None, 
                                                      plot_what='prop_by_dist_of_sig')
# >>>>>> Suppl fig ctrls:
#           x-y-z psf estimates + laser power controls
#           single-spiral controls
#           trig running
speed_stats, _ = analyzeEvoked2P.plot_trig_speed(params=opto_params, expt_type='standard')
#           opsin expression

# %% ===========================================
# ===== Fig 4: evoked timecourse comparison ====
# ==============================================

# fig 4a roi-wise dff timecourse heatmaps
v1_avgs = analyzeEvoked2P.get_avg_trig_responses('V1', params=opto_params, expt_type='standard', resp_type='dff')
m2_avgs = analyzeEvoked2P.get_avg_trig_responses('M2', params=opto_params, expt_type='standard', resp_type='dff')

# fig 4b average timecourse

# fig 4c neuron heatmaps

# fig 4d response time distributions
_, _ = analyzeEvoked2P.plot_response_stats_comparison(params=opto_params, 
                                                      expt_type='standard', 
                                                      resp_type='dff', 
                                                      which_neurons='non_stimd', 
                                                      response_stats=full_resp_stats, 
                                                      axis_handle=None, 
                                                      plot_what='response_time')


# fig 4e? : for each fov, Compare tau of post stim decay to predicted tau from eigenvalue of xcorr mat

# fig 4f: sequence xval

# %% ======================================
# ===== Fig 5: PCA trial trajectories =====
# =========================================


# %% ==============================================
# ===== Fig 6: responses vs. spont timescales =====
# =================================================

# >>>>>> opto vs tau, including dyanmics of that