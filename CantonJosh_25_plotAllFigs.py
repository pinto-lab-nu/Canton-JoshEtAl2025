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
# [LYN INSERT HERE]

# fig 1g regression results 
# calling this method will either run or save, but not recommended. for bash script see /[path]
# [LYN INSERT HERE]

# === Fig S1: tau regression simulations ===
# [LYN INSERT HERE]

# === Fig S2: merfish regression controls ===
# [LYN INSERT HERE]

# %% =========================================
# ===== Fig 2: spontaneous 2p timescales =====
# ============================================

# fig2_handle, fig2_data = plot_spont_tau(params=tau_params)

fig_handle = plt.plot()

# fig 2a: FOV and cranial window egs with raw traces
ax1 = plt.subplot(231)

# fig 2b: 2p tau summary
ax2 = plt.subplot(232)
v1_taus, v1_keys, v1_total = analyzeSpont2P.get_all_tau('V1', params = tau_params, dff_type = 'residuals_dff')
m2_taus, m2_keys, m2_total = analyzeSpont2P.get_all_tau('M2', params = tau_params, dff_type = 'residuals_dff')
tau_stats, ax_tau = analyzeSpont2P.plot_area_tau_comp(v1_taus=v1_taus, m2_taus=m2_taus, axis_handle = ax2, params = tau_params)
tau_stats['V1_total_num_cells'] = v1_total
tau_stats['M2_total_num_cells'] = m2_total

# fig 2c: clustering FOV example
ax3a = plt.subplot(223)
ax3b = plt.subplot(223)
v1_centr, v1_rec_ids = analyzeSpont2P.get_centroids_by_rec(v1_keys)
m2_centr, m2_rec_ids = analyzeSpont2P.get_centroids_by_rec(m2_keys)

fov_v1 = analyzeSpont2P.plot_tau_fov(v1_keys, v1_rec_ids, which_sess=2, do_zscore=False, prctile_cap=[0,95], axis_handle = ax3a)
fov_m2 = analyzeSpont2P.plot_tau_fov(v1_keys, v1_rec_ids, which_sess=17, do_zscore=False, prctile_cap=[0,95], axis_handle = ax3b)
# good ones m2: 0, 4, 5, 10 (95th prct), 17, 22
# good ones v1: 0 , 2 

# fig 2d: clustering
ax4 = plt.subplot(224)
clust_stats_v1 , tau_diff_mat_v1 = analyzeSpont2P.clustering_by_tau(v1_taus, v1_centr, v1_rec_ids, params = tau_params)
clust_stats_m2 , tau_diff_mat_m2 = analyzeSpont2P.clustering_by_tau(m2_taus, m2_centr, m2_rec_ids, params = tau_params)
cax = analyzeSpont2P.plot_clustering_comp(v1_clust=clust_stats_v1,m2_clust=clust_stats_m2, params = tau_params)


# === Fig S3: taus controls ===
# taus without glm and on deconvolved data
tau_stats_noregr, _ = analyzeSpont2P.plot_area_tau_comp(axis_handle = ax, params = tau_params, dff_type = 'noGlm_dff')
tau_stats_deconv, _ = analyzeSpont2P.plot_area_tau_comp(axis_handle = ax, params = tau_params, dff_type = 'residuals_deconv')

# clustering just on low or high tau cells
low_tau_params  = deepcopy(tau_params)
high_tau_params = deepcopy(tau_params)
low_tau_params['max_tau']  = np.median(v1_taus)
high_tau_params['min_tau'] = np.median(v1_taus)
clust_stats_v1_lowtau,_  = analyzeSpont2P.clustering_by_tau(v1_taus, v1_centr, v1_rec_ids, params = low_tau_params)
clust_stats_v1_hightau,_ = analyzeSpont2P.clustering_by_tau(v1_taus, v1_centr, v1_rec_ids, params = high_tau_params)
low_tau_params['max_tau']  = np.median(m2_taus)
high_tau_params['min_tau'] = np.median(m2_taus)
clust_stats_m2_lowtau,_  = analyzeSpont2P.clustering_by_tau(m2_taus, m2_centr, m2_rec_ids, params = low_tau_params)
clust_stats_m2_hightau,_ = analyzeSpont2P.clustering_by_tau(m2_taus, m2_centr, m2_rec_ids, params = high_tau_params)

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
# >>>>> Plot sig and non sig mag together 
# (could edit this function to take an additional dictionary, and plot with shade colors if so)
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

# === Fig S4: 2p opto controls ===

# x-y-z psf estimates
# [NETO INSERT HERE]

# laser power controls
# [NETO INSERT HERE]

# single-spiral controls
resp_prob_summary_short_stim, _ = analyzeEvoked2P.plot_response_stats_comparison(params=opto_params, 
                                                                                expt_type='short_stim', 
                                                                                resp_type='dff', 
                                                                                axis_handle=None, 
                                                                                plot_what='response_probability')

# trig running
speed_stats, _ = analyzeEvoked2P.plot_trig_speed(params=opto_params, expt_type='standard')

# opsin expression
expression_stats, _ = analyzeEvoked2P.plot_opsin_expression_vs_response(params=opto_params, 
                                                                         expt_type='standard', 
                                                                         resp_type='dff',
                                                                         plot_what='stimd', 
                                                                         axis_handle=None)
_, _ = analyzeEvoked2P.plot_opsin_expression_vs_response(params=opto_params, 
                                                        expt_type='standard', 
                                                        resp_type='dff',
                                                        plot_what='non_stimd', 
                                                        v1_data=expression_stats['V1_summary'],
                                                        m2_data=expression_stats['M2_summary'],
                                                        axis_handle=None)

# %% ===========================================
# ===== Fig 4: evoked timecourse comparison ====
# ==============================================

# fig 4a roi-wise dff timecourse heatmaps
v1_avgs = analyzeEvoked2P.get_avg_trig_responses('V1', params=opto_params, expt_type='standard', resp_type='dff')
m2_avgs = analyzeEvoked2P.get_avg_trig_responses('M2', params=opto_params, expt_type='standard', resp_type='dff')

# fig 4b average timecourse

# fig 4c neuron heatmaps over time

# fig 4d response time distributions
_, _ = analyzeEvoked2P.plot_response_stats_comparison(params=opto_params, 
                                                      expt_type='standard', 
                                                      resp_type='dff', 
                                                      which_neurons='non_stimd', 
                                                      response_stats=full_resp_stats, 
                                                      axis_handle=None, 
                                                      plot_what='response_time')


# fig 4e: sequence xval

# === Fig S6: timing with deconvolved traces ===


# %% ======================================
# ===== Fig 5: PCA trial trajectories =====
# =========================================


# %% ==============================================
# ===== Fig 6: responses vs. spont timescales =====
# =================================================

# >>>>>> opto vs tau, including dyanmics of that


# %% ==============================================
# =================================================
# FIGURE SOURCE CODE

# def plot_spont_tau(params=tau_params):
    
#     return fig_handle, fig_data