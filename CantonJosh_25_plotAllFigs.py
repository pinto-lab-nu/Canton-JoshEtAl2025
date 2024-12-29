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
# >>> TO DO

# fig 1c raw data with running plus exponential egs
# >>> TO DO

# fig 1d widefield tau map
# >>> TO DO

# fig 1e tau area summary
# >>> TO DO

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

# we should preferably create methods at the bottom for each figure and call them from here
# makes it a lot easier to follow what's going on
# fig2_handle, fig2_data = plot_spont_tau(params=tau_params)

# generate layout here and pass axis handles 
fig_handle = plt.plot()

# fig 2a: FOV and cranial window egs with raw traces for each area
ax1 = plt.subplot(231)
# >>> TO DO

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

# generate layout here and pass axis handles 
fig3_params = {'sess_eg_v1' : 0,
               'stim_eg_v1' : 1,
               'sess_eg_m2' : 0,
               'stim_eg_m2' : 1}

# fig 3a raw evoked traces and/or pixel-wise dff example (or suppl movie)
# >>> TO DO

# fig 3b area eg response map
# >>>>>>>>>>> still need to pick best which_sess and which_stim
_, _ = analyzeEvoked2P.plot_resp_fov('V1', 
                                     which_sess=fig3_params['sess_eg_v1'], 
                                     which_stim=fig3_params['stim_eg_v1'], 
                                     expt_type='standard', 
                                     resp_type='dff', 
                                     plot_what='peak_mag', 
                                     prctile_cap=[0,98], 
                                     signif_only=False, 
                                     highlight_signif=True, 
                                     axis_handle=None)
_, _ = analyzeEvoked2P.plot_resp_fov('M2', 
                                     which_sess=fig3_params['sess_eg_m2'], 
                                     which_stim=fig3_params['stim_eg_m2'], 
                                     expt_type='standard', 
                                     resp_type='dff', 
                                     plot_what='peak_mag', 
                                     prctile_cap=[0,98], 
                                     signif_only=False, 
                                     highlight_signif=True, 
                                     axis_handle=None)

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
                                                                    plot_what='response_magnitude',
                                                                    signif_only=True, 
                                                                    overlay_non_sig=True)

# fig 3e response vs distance
_, _ = analyzeEvoked2P.plot_response_stats_comparison(params=opto_params, 
                                                      expt_type='standard', 
                                                      resp_type='dff', 
                                                      which_neurons='non_stimd', 
                                                      response_stats=full_resp_stats, 
                                                      axis_handle=None, 
                                                      plot_what='prop_by_dist_of_sig',
                                                      signif_only=True, 
                                                      overlay_non_sig=False)

# === Fig S4: 2p opto controls ===

# generate layout here and pass axis handles 

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

fig4_params = {'sess_eg_v1' : 0,
               'stim_eg_v1' : 1,
               'sess_eg_m2' : 0,
               'stim_eg_m2' : 1}

# get data
v1_avgs = analyzeEvoked2P.get_avg_trig_responses('V1', params=opto_params, expt_type='standard', resp_type='dff')
m2_avgs = analyzeEvoked2P.get_avg_trig_responses('M2', params=opto_params, expt_type='standard', resp_type='dff')

# generate layout here and pass axis handles ; params

# fig 4a roi-wise dff timecourse heatmaps
# >>>>>>>>>>> still need to pick best which_sess and which_stim
_, _ = analyzeEvoked2P.plot_resp_fov('V1', 
                                     which_sess=fig4_params['sess_eg_v1'], 
                                     which_stim=fig4_params['stim_eg_v1'], 
                                     expt_type='standard', 
                                     resp_type='dff', 
                                     plot_what='full_seq', 
                                     prctile_cap=[0,98], 
                                     signif_only=False, 
                                     highlight_signif=True, 
                                     axis_handle=None)
_, _ = analyzeEvoked2P.plot_resp_fov('M2', 
                                     which_sess=fig4_params['sess_eg_m2'], 
                                     which_stim=fig4_params['stim_eg_m2'], 
                                     expt_type='standard', 
                                     resp_type='dff', 
                                     plot_what='full_seq', 
                                     prctile_cap=[0,98], 
                                     signif_only=False, 
                                     highlight_signif=True, 
                                     axis_handle=None)

# fig 4b average timecourse
# these look different than Neto's need to figure out why. 
# maybe here we should plot the grand average regardless of significance
_, _, _, = analyzeEvoked2P.plot_response_grand_average(params=opto_params, 
                                                       expt_type='standard', 
                                                       resp_type='dff', 
                                                       signif_only=True, 
                                                       which_neurons='non_stimd', 
                                                       v1_data=v1_avgs, 
                                                       m2_data=m2_avgs, 
                                                       axis_handle=None, 
                                                       norm_type='peak')
    
# fig 4c neuron heatmaps over time. 
# these look different than Neto's need to figure out why
_, _, _, = analyzeEvoked2P.plot_avg_response_heatmap('V1', 
                                                     params=opto_params, 
                                                     expt_type='standard', 
                                                     resp_type='dff', 
                                                     signif_only=True, 
                                                     which_neurons='non_stimd', 
                                                     avg_data=v1_avgs, 
                                                     axis_handle=None, 
                                                     fig_handle=None, 
                                                     norm_type='minmax')
_, _, _, = analyzeEvoked2P.plot_avg_response_heatmap('M2', 
                                                     params=opto_params, 
                                                     expt_type='standard', 
                                                     resp_type='dff', 
                                                     signif_only=True, 
                                                     which_neurons='non_stimd', 
                                                     avg_data=m2_avgs, 
                                                     axis_handle=None, 
                                                     fig_handle=None, 
                                                     norm_type='minmax')

# fig 4d response time distributions
_, _ = analyzeEvoked2P.plot_response_stats_comparison(params=opto_params, 
                                                      expt_type='standard', 
                                                      resp_type='dff', 
                                                      which_neurons='non_stimd', 
                                                      response_stats=full_resp_stats, 
                                                      axis_handle=None, 
                                                      plot_what='response_time',
                                                      signif_only=True)


# fig 4e: sequence xval
_, _, xval_results = analyzeEvoked2P.plot_trial_xval(area='M2', 
                                                     params=opto_params, 
                                                     expt_type='high_trial_count', 
                                                     resp_type='dff', 
                                                     signif_only=True, 
                                                     which_neurons='non_stimd', 
                                                     axis_handle=None, 
                                                     fig_handle=None)

# === Fig S6: timing with deconvolved traces ===
v1_avgs_deconv = analyzeEvoked2P.get_avg_trig_responses('V1', params=opto_params, expt_type='standard', resp_type='deconv')
m2_avgs_deconv = analyzeEvoked2P.get_avg_trig_responses('M2', params=opto_params, expt_type='standard', resp_type='deconv')
full_resp_stats_deconv = analyzeEvoked2P.compare_response_stats(params=opto_params, 
                                                                expt_type='standard', 
                                                                resp_type='deconv', 
                                                                which_neurons='non_stimd', 
                                                                signif_only=True)
# heatmaps
_, _, _, = analyzeEvoked2P.plot_avg_response_heatmap('V1', 
                                                     params=opto_params, 
                                                     expt_type='standard', 
                                                     resp_type='deconv', 
                                                     signif_only=True, 
                                                     which_neurons='non_stimd', 
                                                     avg_data=v1_avgs_deconv, 
                                                     axis_handle=None, 
                                                     fig_handle=None, 
                                                     norm_type='minmax')
_, _, _, = analyzeEvoked2P.plot_avg_response_heatmap('M2', 
                                                     params=opto_params, 
                                                     expt_type='standard', 
                                                     resp_type='deconv', 
                                                     signif_only=True, 
                                                     which_neurons='non_stimd', 
                                                     avg_data=m2_avgs_deconv, 
                                                     axis_handle=None, 
                                                     fig_handle=None, 
                                                     norm_type='minmax')

# response time distributions
_, _ = analyzeEvoked2P.plot_response_stats_comparison(params=opto_params, 
                                                      expt_type='standard', 
                                                      resp_type='deconv', 
                                                      which_neurons='non_stimd', 
                                                      response_stats=full_resp_stats_deconv, 
                                                      axis_handle=None, 
                                                      plot_what='response_time',
                                                      signif_only=True)

# %% ======================================
# ===== Fig 5: PCA trial trajectories =====
# =========================================

# generate layout here and pass axis handles 

# run data analysis 
v1_pca_results = analyzeEvoked2P.batch_trial_pca('V1', 
                                                params=opto_params, 
                                                expt_type='standard', 
                                                resp_type='dff')
m2_pca_results = analyzeEvoked2P.batch_trial_pca('M2', 
                                                params=opto_params, 
                                                expt_type='standard+high_trial_count', 
                                                resp_type='dff')

# fig 5a trajectory examples for each area:  
# [NETO TO DO]
# area_pca_resluts above already have the trajectories
# just a matter of writing a method that takes an exaple and plots 3d trajectories

# fig 5b baseline x response distance for V1 and M2
_, _ = analyzeEvoked2P.plot_pca_dist_scatter('V1', 
                                             params=opto_params, 
                                             expt_type='standard', 
                                             resp_type='dff', 
                                             eg_ids=None, 
                                             trial_pca_results=v1_pca_results, 
                                             axis_handle=None)
_, _ = analyzeEvoked2P.plot_pca_dist_scatter('M2', 
                                             params=opto_params, 
                                             expt_type='standard+high_trial_count', 
                                             resp_type='dff', 
                                             eg_ids=None, 
                                             trial_pca_results=m2_pca_results, 
                                             axis_handle=None)

# fig 5c comparison of V1 and M2 (single-expt distributions of correlations / pvals)
# [NETO TO DO]
# here I was just thinking to write a simple method to plot simple histograms of 
# corr coefficients across experiments for the two area and compare statistically
# these are already collected in area_pca_results['corr_by_expt'] and ['p_by_expt']

# %% ==============================================
# ===== Fig 6: responses vs. spont timescales =====
# =================================================

# generate layout here and pass axis handles 

# related to the timing differences in fig 4 (compared to previous versions):
# no differences here, given that most cells peak very late. need to figure this out 
# fig 6a peak time vs tau
_, _, tau_vs_opto_comp_summary = analyzeEvoked2P.plot_opto_vs_tau_comparison(area=None, 
                                                                             plot_what='peak_time', 
                                                                             params=opto_params, 
                                                                             expt_type='standard', 
                                                                             resp_type='dff', 
                                                                             dff_type='residuals_dff', 
                                                                             tau_vs_opto_comp_summary=None, 
                                                                             axis_handles=None)

# fig 6b peak magnitude
_, _, _ = analyzeEvoked2P.plot_opto_vs_tau_comparison(area=None, 
                                                    plot_what='peak_mag', 
                                                    params=opto_params, 
                                                    expt_type='standard', 
                                                    resp_type='dff', 
                                                    dff_type='residuals_dff', 
                                                    tau_vs_opto_comp_summary=tau_vs_opto_comp_summary, 
                                                    axis_handles=None)

# fig 6c maybe, peak width: I forgot to write that into the dj tables. 
# will need to be coded posthoc and added as a category to:
#    - analyzeEvoked2P.tau_vs_opto_comp_summary
#    - analyzeEvoked2P.opto_vs_tau

# fig 6d overall tau vs stimd/reponding probability
# (this expects a list of two axis handles)
_, _, _ = analyzeEvoked2P.plot_opto_vs_tau_comparison(area=None, 
                                                    plot_what='prob', 
                                                    params=opto_params, 
                                                    expt_type='standard', 
                                                    resp_type='dff', 
                                                    dff_type='residuals_dff', 
                                                    tau_vs_opto_comp_summary=tau_vs_opto_comp_summary, 
                                                    axis_handles=None)

# fig 6e, maybe evolution of tau vs stimd/reponding probability over time
# (this expects a list of 2 x 10 axis handles)
_, _, _ = analyzeEvoked2P.plot_opto_vs_tau_comparison(area=None, 
                                                    plot_what='prob_by_time_by_overall', 
                                                    params=opto_params, 
                                                    expt_type='standard', 
                                                    resp_type='dff', 
                                                    dff_type='residuals_dff', 
                                                    tau_vs_opto_comp_summary=tau_vs_opto_comp_summary, 
                                                    axis_handles=None)


# %% ==============================================
# =================================================
# FIGURE SOURCE CODE
# >>> TO DO: move scripts from top under fig-specific methods
# def plot_spont_tau(params=tau_params):
    
#     return fig_handle, fig_data