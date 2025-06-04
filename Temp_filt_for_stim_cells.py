# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:29:41 2025

@author: jec822
"""
import numpy as np
from scipy.signal import medfilt
from scipy.stats import zscore

def process_trig_dff_trials(a, kernel_size=5, z_score=False):
    peak_values = []
    auc_values = []

    for i, trace in enumerate(a['trig_dff_trials']):
        # Skip if trace is invalid
        if trace is None or len(trace) < 121 or np.all(np.isnan(trace)):
            peak_values.append(np.nan)
            auc_values.append(np.nan)
            continue

        # Optional z-scoring
        if z_score:
            trace = zscore(trace, nan_policy='omit')

        # Apply median filter to the trace starting at index 100
        filtered = medfilt(trace[100:], kernel_size=kernel_size)

        # Normalize to have min = 0
        filtered = filtered - np.nanmin(filtered)

        # Calculate peak (mean between indices 100 and 120 inclusive => 0:21 in sliced trace)
        peak = np.nanmean(filtered[0:21])

        # Calculate AUC from index 100 to end (which is the full filtered slice)
        auc = np.nansum(filtered)

        peak_values.append(peak)
        auc_values.append(auc)

    return np.array(peak_values), np.array(auc_values)


# %%

import numpy as np
from collections import defaultdict

def group_peaks_by_stim_and_roi(peak_array, roi_ids, stim_ids, threshold):
    grouped_peaks = defaultdict(list)
    proportions_dict = {}

    # Step 1: Group peak values by (stim_id, roi_id)
    for peak, roi, stim in zip(peak_array, roi_ids, stim_ids):
        key = (stim, roi)
        grouped_peaks[key].append(peak)

    # Step 2: Compute proportions above threshold per group
    for key, values in grouped_peaks.items():
        values_np = np.array(values)
        if len(values_np) > 0:
            prop = np.sum(values_np > threshold) / len(values_np)
        else:
            prop = np.nan
        proportions_dict[key] = prop

    # Step 3: Create full-length array where each index gets its group proportion
    full_proportion_array = np.array([
        proportions_dict[(stim, roi)] for roi, stim in zip(roi_ids, stim_ids)
    ])

    return dict(grouped_peaks), proportions_dict, full_proportion_array

# %% single_trial stim only
stim_cells_results_V1=get_single_trial_data(area='V1', params=opto_params, expt_type='standard', resp_type='dff', eg_ids=None, signif_only=False, which_neurons='stimd')
stim_cells_results_M2=get_single_trial_data(area='M2', params=opto_params, expt_type='standard', resp_type='dff', eg_ids=None, signif_only=False, which_neurons='stimd')
# %%

peak_array_m2, auc_array_m2 = process_trig_dff_trials(stim_cells_results_M2,kernel_size=9)
peak_array_v1, auc_array_v1 = process_trig_dff_trials(stim_cells_results_V1,kernel_size=9)

# %%
threshold=2
roi_peaks_v1, proportions_v1_dict,proportions_v1= group_peaks_by_stim_and_roi(peak_array_v1, stim_cells_results_V1['roi_ids'],stim_cells_results_V1['stim_ids'], threshold)
roi_peaks_m2, proportions_m2_dict,proportions_m2 = group_peaks_by_stim_and_roi(peak_array_m2, stim_cells_results_M2['roi_ids'],stim_cells_results_M2['stim_ids'], threshold)

# %%
# peak_array_v1[peak_array_v1 < 2] = 0
# peak_array_m2[peak_array_m2 < 2] = 0


plot_ecdf_comparison(peak_array_v1,peak_array_m2,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='Z-scores of directly stimulated cells',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak z-score",
                     ylabel="Directly stimulated cells")

plot_ecdf_comparison((1-proportions_v1),(1-proportions_m2),
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='Succesful trials in directly stimulated cells',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Proportion of non-responsive trials",
                     ylabel="Directly stimulated cells")
# %% avg stim only
stim_cells_results_M2_avg_stim=analyzeEvoked2P.get_avg_trig_responses('M2', params=opto_params, expt_type='standard', resp_type='dff',which_neurons='stimd')
stim_cells_results_V1_avg_stim=analyzeEvoked2P.get_avg_trig_responses('V1', params=opto_params, expt_type='standard', resp_type='dff',which_neurons='stimd')
# %%
peak_array_mean_V1=stim_cells_results_V1_avg_stim['max_or_min_vals']
peak_array_mean_M2=stim_cells_results_M2_avg_stim['max_or_min_vals']

# peak_array_mean_V1[peak_array_mean_V1 < 2] = 0
# peak_array_mean_M2[peak_array_mean_M2 < 2] = 0



plot_ecdf_comparison(peak_array_mean_V1,peak_array_mean_M2,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='Z-scores of directly stimulated cells',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak z-score",
                     ylabel="Directly stimulated cells")


# %% Single trial sig non stimd cells sig

stim_cells_results_M2_non_stimd=get_single_trial_data(area='M2', params=opto_params, expt_type='standard', resp_type='dff', eg_ids=None, signif_only=False, which_neurons='non_stimd')

stim_cells_results_V1_non_stimd=get_single_trial_data(area='V1', params=opto_params, expt_type='standard', resp_type='dff', eg_ids=None, signif_only=False, which_neurons='non_stimd')
# %%

peak_array_m2, auc_array_m2 = process_trig_dff_trials(stim_cells_results_M2_non_stimd,kernel_size=9)

peak_array_v1, auc_array_v1 = process_trig_dff_trials(stim_cells_results_V1_non_stimd,kernel_size=9)

threshold=2
roi_peaks_v1, proportions_v1_dict,proportions_v1= group_peaks_by_stim_and_roi(peak_array_v1, stim_cells_results_V1_non_stimd['roi_ids'],stim_cells_results_V1_non_stimd['stim_ids'], threshold)

roi_peaks_m2, proportions_m2_dict,proportions_m2 = group_peaks_by_stim_and_roi(peak_array_m2, stim_cells_results_M2_non_stimd['roi_ids'],stim_cells_results_M2_non_stimd['stim_ids'], threshold)

peak_array_v1[peak_array_v1 < 2] = 0
peak_array_m2[peak_array_m2 < 2] = 0
# %%

plot_ecdf_comparison(peak_array_v1,peak_array_m2)



plot_ecdf_comparison((1-proportions_v1),(1-proportions_m2),
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='Proportion of non-responses in non-stimd cells',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Proportion of non-responsive trials",
                     ylabel="Non-stimulated cells")
# %% avg trial sig non stimd cells all

stim_cells_results_M2_avg=analyzeEvoked2P.get_avg_trig_responses('M2', params=opto_params, expt_type='standard', resp_type='dff',which_neurons='non_stimd',signif_only=False)
stim_cells_results_V1_avg=analyzeEvoked2P.get_avg_trig_responses('V1', params=opto_params, expt_type='standard', resp_type='dff',which_neurons='non_stimd',signif_only=False)

# %% avg trial sig non stimd cells sig
stim_cells_results_M2_avg_sig=analyzeEvoked2P.get_avg_trig_responses('M2', params=opto_params, expt_type='standard', resp_type='dff',which_neurons='non_stimd',signif_only=True)
stim_cells_results_V1_avg_sig=analyzeEvoked2P.get_avg_trig_responses('V1', params=opto_params, expt_type='standard', resp_type='dff',which_neurons='non_stimd',signif_only=True)

# %%
peak_array_mean_V1=stim_cells_results_V1_avg_sig['max_or_min_vals']
peak_array_mean_M2=stim_cells_results_M2_avg_sig['max_or_min_vals']

plot_ecdf_comparison(peak_array_mean_V1,peak_array_mean_M2)