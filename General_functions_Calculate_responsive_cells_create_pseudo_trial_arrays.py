# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 16:19:11 2025

@author: jec822
"""
from utils import connect_to_dj
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
VM = connect_to_dj.get_virtual_modules()


# ======= Local modules ========
# code_dir = "/Users/lpr6177/Documents/code/Canton-JoshEtAl2025/"
code_dir = "/Users/Jec822/Documents/GitHub/Canton-JoshEtAl2025"
sys.path.insert(0,code_dir)
from analyzeSpont2P import params as tau_params
import analyzeSpont2P
from analyzeEvoked2P import params as opto_params
import analyzeEvoked2P
import Canton_Josh_et_al_2025_analysis_plotting_functions as analysis_plotting_functions

# %%
import numpy as np
from collections import defaultdict

def group_peaks_by_roi_and_stim(peak_array, roi_ids, stim_ids, threshold, max_trials_per_group=10):
    grouped_peaks = defaultdict(list)
    temp_grouping = defaultdict(list)

    # Step 1: Temporarily collect all peak values per group
    for peak, roi, stim in zip(peak_array, roi_ids, stim_ids):
        key = (roi, stim)
        temp_grouping[key].append(peak)

    # Step 2: Enforce max_trials_per_group if specified
    for key, peaks in temp_grouping.items():
        if max_trials_per_group is not None and len(peaks) > max_trials_per_group:
            selected_peaks = peaks[:max_trials_per_group]
        else:
            selected_peaks = peaks
        grouped_peaks[key] = selected_peaks

    # Step 3: Compute proportions above threshold per group
    proportions_dict = {}
    for key, values in grouped_peaks.items():
        values_np = np.array(values)
        if len(values_np) > 0:
            prop = np.sum(values_np > threshold) / len(values_np)
        else:
            prop = np.nan
        proportions_dict[key] = prop

    # Step 4: Create full-length array where each index gets its group proportion
    full_proportion_array = np.array([
        proportions_dict.get((roi, stim), np.nan) for roi, stim in zip(roi_ids, stim_ids)
    ])

    return dict(grouped_peaks), proportions_dict, full_proportion_array

# %%
import pandas as pd

def group_peaks_by_roi_and_stim_df(peak_array, roi_ids, stim_ids, threshold, max_trials_per_group=10):
    df = pd.DataFrame({'roi': roi_ids, 'stim': stim_ids, 'peak': peak_array})
    
    # Sort if needed
    df.sort_values(['roi', 'stim'], inplace=True)

    # Apply max_trials_per_group
    df = df.groupby(['roi', 'stim']).head(max_trials_per_group)

    # Compute proportions
    proportions = df.groupby(['roi', 'stim'])['peak'].apply(lambda x: np.mean(x > threshold)).to_dict()

    # Full array
    full_proportion_array = [
        proportions.get((r, s), np.nan) for r, s in zip(roi_ids, stim_ids)
    ]

    # Also group peaks
    grouped_peaks = df.groupby(['roi', 'stim'])['peak'].apply(list).to_dict()

    return grouped_peaks, proportions, np.array(full_proportion_array)



# %%
import numpy as np
from collections import defaultdict

def group_dicts_by_roi_and_stim(dict_list, roi_ids, stim_ids, max_trials_per_group=None):
    """
    Groups dictionaries by (roi_id, stim_id) pair with optional limit on trials per group.

    Parameters
    ----------
    dict_list : list of dict
        Each item is a dictionary containing trial-wise features.
    roi_ids : array-like
        List or array of ROI identifiers, same length as dict_list.
    stim_ids : array-like
        List or array of Stim identifiers, same length as dict_list.
    max_trials_per_group : int, optional
        Maximum number of trials to include per (roi_id, stim_id) group.

    Returns
    -------
    grouped_dict : dict
        Keys are (roi_id, stim_id), values are lists of dicts.
    """
    temp_grouping = defaultdict(list)
    grouped = defaultdict(list)

    # Step 1: Group all entries
    for d, roi, stim in zip(dict_list, roi_ids, stim_ids):
        key = (roi, stim)
        temp_grouping[key].append(d)

    # Step 2: Enforce max_trials_per_group if specified
    for key, items in temp_grouping.items():
        if max_trials_per_group is not None:
            grouped[key] = items[:max_trials_per_group]
        else:
            grouped[key] = items

    return dict(grouped)


# %%
def get_std_com_above_threshold(roi_com_times_dict, proportions_dict, threshold):
    std_results = {}
    for key in roi_com_times_dict:
        proportion = proportions_dict.get(key, 0)
        if proportion > threshold:
            com_times = roi_com_times_dict[key]
            if len(com_times) > 0:
                std_results[key] = np.nanstd(com_times)
            else:
                std_results[key] = np.nan
        else:
            std_results[key] = np.nan  # Optional: mark keys below threshold
    return std_results

# %%
def get_avg_com_above_threshold(roi_com_times_dict, proportions_dict, threshold):
    std_results = {}
    for key in roi_com_times_dict:
        proportion = proportions_dict.get(key, 0)
        if proportion > threshold:
            com_times = roi_com_times_dict[key]
            if len(com_times) > 0:
                std_results[key] = np.nanmean(com_times)
            else:
                std_results[key] = np.nan
        else:
            std_results[key] = np.nan  # Optional: mark keys below threshold
    return std_results

# %%
import numpy as np

def filter_dict_by_threshold(data_dict, proportions_dict, min_threshold=None, max_threshold=None):
    """
    Return a filtered version of data_dict where the corresponding value in proportions_dict
    is within the specified min/max range and is not NaN.

    Parameters:
        data_dict (dict): Dictionary with data to be filtered.
        proportions_dict (dict): Dictionary with proportion values for filtering.
        min_threshold (float, optional): Minimum proportion (inclusive). If None, no lower bound.
        max_threshold (float, optional): Maximum proportion (inclusive). If None, no upper bound.

    Returns:
        dict: Filtered version of data_dict.
    """
    filtered = {}
    for key in data_dict:
        val = proportions_dict.get(key, None)
        if val is None or np.isnan(val):
            continue
        if min_threshold is not None and val < min_threshold:
            continue
        if max_threshold is not None and val > max_threshold:
            continue
        filtered[key] = data_dict[key]
    return filtered

# %%
import numpy as np

def average_arrays_within_dict(nested_array_dict):
    """
    For each key in the input dictionary, stack the list of 1D arrays and return the mean array.

    Parameters:
        nested_array_dict (dict): Dictionary with lists of 1D NumPy arrays as values.

    Returns:
        dict: Dictionary with the same keys, where each value is the average 1D array.
    """
    averaged_dict = {}
    for key, array_list in nested_array_dict.items():
        # Stack and average only if the list is non-empty
        if len(array_list) > 0:
            stacked = np.vstack(array_list)
            averaged = np.nanmean(stacked, axis=0)
            averaged_dict[key] = averaged
        else:
            averaged_dict[key] = np.array([])  # or np.nan, depending on preference
    return averaged_dict

# %%
def dict_of_arrays_to_2d_array_padded(array_dict, fill_value=np.nan):
    """
    Convert a dictionary of 1D arrays into a 2D array, padding shorter arrays with NaNs.

    Parameters:
        array_dict (dict): Dictionary where each value is a 1D NumPy array.
        fill_value: Value used to pad shorter arrays (default is np.nan).

    Returns:
        np.ndarray: 2D NumPy array with padded rows.
    """
    max_len = max(len(arr) for arr in array_dict.values())
    padded_arrays = []

    for arr in array_dict.values():
        padded = np.full(max_len, fill_value)
        padded[:len(arr)] = arr  # pad the end
        padded_arrays.append(padded)

    return np.vstack(padded_arrays)

# %%
import numpy as np
import pandas as pd

def generate_pseudo_trials(
    trial_data,
    n_iters=1,
    baseline_window=(5000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80
):
    """
    Generate real and pseudo-trials for calcium traces, grouped by ROI and stim.

    Parameters
    ----------
    trial_data : dict or pd.DataFrame-like
        Must contain keys/columns: 'roi_ids', 'stim_ids', 'roi_keys', 'trig_dff_trials', 'time_axis_sec'
    n_iters : int
        Number of iterations of pseudo-trial generation
    baseline_window : tuple
        Index window to extract baseline DFF trace from (start_idx, end_idx)
    trial_crop_idx : tuple or None
        (start, end) index to crop trials to; if None, will use shortest trial length
    baseline_start_idx : int
        Start index within baseline_dff for extracting pseudo-trials
    max_offset : int
        Max offset for pseudo-trial generation
    zscore_to_baseline : bool
        Whether to z-score pseudo-trials to baseline
    baseline_window_for_z : int
        Number of samples used to compute z-score mean/std

    Returns
    -------
    real_trial_dffs : list of np.ndarray
        Real DFF trials
    pseudo_trial_dffs : list of np.ndarray
        Flattened list of all pseudo-trial replicates
    pseudo_dfs_iteration : dict
        Dictionary of iteration DataFrames: {iter_0: df, iter_1: df, ...}
    """

    # Extract relevant arrays
    roi_ids = np.array(trial_data['roi_ids'])
    stim_ids = np.array(trial_data['stim_ids'])
    roi_keys_all = np.array(trial_data['roi_keys'])
    trials = trial_data['trig_dff_trials']
    # t_axes_all = np.array(trial_data['time_axis_sec'], dtype=object)
    unique_rois = np.unique(roi_ids)

    # Determine min trial length if not provided
    if trial_crop_idx is None:
        min_len = min(len(trial) for trial in trials)
    else:
        min_len = trial_crop_idx[1] - trial_crop_idx[0]

    # Truncate all trials
    trial_dffs_all = np.array([trial[:min_len] for trial in trials])
    # t_axes_all = np.array([ax[:min_len] for ax in t_axes_all])

    # Output containers
    # real_trial_dffs = []         # all real trials (flattened list of arrays)

    pseudo_trial_dffs = []
    pseudo_dfs_iteration = {}

    for i in range(n_iters):
        pseudo_dffs_list = []

        for roi in unique_rois:
            ridx = roi_ids == roi
            roi_keys = roi_keys_all[ridx]
            stim_set = stim_ids[ridx]
            roi_key = roi_keys[0]
            stim_list = np.unique(stim_set)

            for stim in stim_list:
                tidx = np.where((roi_ids == roi) & (stim_ids == stim))[0]
                if len(tidx) == 0:
                    continue
               
                # Real trials
                real_trials = trial_dffs_all[tidx]
                # real_trial_dffs.extend(real_trials)
               
                # Fetch baseline from DataJoint (assumes access to VM['twophoton'])
                baseline_dff = (VM['twophoton'].Rawf2P & roi_key).fetch1('rawf')[0][baseline_window[0]:baseline_window[1]]
                # baseline_dff = (VM['twophoton'].Dff2P & roi_key).fetch1('dff')[0][baseline_window[0]:baseline_window[1]]

                pseudo_trials = analyzeEvoked2P.create_pseudo_trials_from_baseline(
                    baseline_dff=baseline_dff,
                    number_of_trials=len(real_trials),
                    trial_length=min_len,
                    baseline_start_idx=baseline_start_idx,
                    max_offset=max_offset,
                    zscore_to_baseline=zscore_to_baseline,
                    within_trial_baseline_frames=baseline_window_for_z,
                    bootstrap=False,
                    min_spacing_between_starts=100
                )
                pseudo_trial_dffs.extend(pseudo_trials)
                pseudo_dffs_list.extend(pseudo_trials)

        # Store in a DataFrame per iteration
        pseudo_dfs_iteration[f"iter_{i}"] = pd.DataFrame(pseudo_dffs_list)

    return pseudo_dfs_iteration

# %% This is use to genearate pseudo dataset from responsive cells using their own baseline period

import numpy as np
import pandas as pd

def generate_pseudo_trials_select_cells(
    roi_ids,
    stim_ids,
    roi_keys_all,
    trials,
    n_iters=1,
    baseline_window=(5000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80,
    min_spacing_between_starts=100,
    repeat_count=10
):
    """
    Generate pseudo-trials from baseline calcium activity.

    Returns
    -------
    pseudo_dfs_iteration : dict
        Dictionary of iteration DataFrames containing pseudo-trials and metadata.
    """
    
    trials = trials.reset_index(drop=True)
    unique_rois, unique_indices = np.unique(roi_ids, return_index=True)

    # Subset all relevant data based on unique indices
    stim_ids = stim_ids[unique_indices]
    roi_keys = roi_keys_all[unique_indices]
    trials = trials[unique_indices]
    
    stim_types = list(range(1, repeat_count + 1))

    pseudo_trial_dffs = []
    pseudo_dfs_iteration = {}

    stim_ids_repeated = np.repeat(stim_types, len(unique_rois))
    
    roi_ids_repeated = np.concatenate([unique_rois for _ in range(repeat_count)])
    roi_keys_repeated = np.concatenate([roi_keys for _ in range(repeat_count)])
    trials_repeated = pd.concat([trials.copy() for _ in range(repeat_count)], ignore_index=True)

    pseudo_dfs_iteration = {}

    for i in range(n_iters):
        metadata_list = []

        for roi in unique_rois:
            ridx = roi_ids_repeated == roi
            stim_set = stim_ids_repeated[ridx]
            roi_key = roi_keys_repeated[ridx][0][0]
            stim_list = np.unique(stim_set)
            
            real_trials = trials[ridx[0:len(unique_rois)]]
            real_trials = real_trials.reset_index(drop=True)
            min_len = len(real_trials[0][0])
            # print(roi)
            for stim in stim_list:
                
                # Get baseline trace
                baseline_dff = (VM['twophoton'].Rawf2P & roi_key).fetch1('rawf')[0][baseline_window[0]:baseline_window[1]]
    
                pseudo_trials = analyzeEvoked2P.create_pseudo_trials_from_baseline(
                    baseline_dff=baseline_dff,
                    number_of_trials=len(real_trials[0]),
                    trial_length=min_len,
                    baseline_start_idx=baseline_start_idx,
                    max_offset=max_offset,
                    zscore_to_baseline=zscore_to_baseline,
                    within_trial_baseline_frames=baseline_window_for_z,
                    bootstrap=False,
                    min_spacing_between_starts=min_spacing_between_starts
                )
    
                for trial in pseudo_trials:
                    metadata_list.append({
                        'roi_id': roi,
                        'stim_id': stim,
                        'roi_key': roi_key,
                        'trials_arrays': trial
                    })

        # Convert list of dicts to DataFrame
        df_iter = pd.DataFrame(metadata_list)
        pseudo_dfs_iteration[f"iter_{i}"] = df_iter

    return pseudo_dfs_iteration

# %%
import numpy as np
import pandas as pd
import random

def generate_pseudo_trials_subsampled(
    trial_data,
    n_iters=1000,
    n_sampled_rois=200,
    baseline_window=(5000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80
):
    """
    Generate pseudo-trials for a random subset of ROIs.

    Parameters
    ----------
    trial_data : dict
        Must contain 'roi_ids', 'stim_ids', 'roi_keys', 'trig_dff_trials', 'time_axis_sec'
    n_iters : int
        Number of pseudo-trial replicates to generate
    n_sampled_rois : int
        Number of ROIs to sample per iteration
    [Other parameters same as before...]

    Returns
    -------
    pseudo_dfs_iteration : dict
        {iter_0: DataFrame of pseudo trials, ...}
    """

    roi_ids = np.array(trial_data['roi_ids'])
    stim_ids = np.array(trial_data['stim_ids'])
    roi_keys_all = np.array(trial_data['roi_keys'])
    trials = trial_data['trig_dff_trials']
    t_axes_all = np.array(trial_data['time_axis_sec'], dtype=object)
    unique_rois = np.unique(roi_ids)

    # Determine trial length
    if trial_crop_idx is None:
        min_len = min(len(trial) for trial in trials)
    else:
        min_len = trial_crop_idx[1] - trial_crop_idx[0]

    trial_dffs_all = np.array([trial[:min_len] for trial in trials])
    t_axes_all = np.array([ax[:min_len] for ax in t_axes_all])

    # Output container
    pseudo_dfs_iteration = {}

    for i in range(n_iters):
        sampled_rois = np.random.choice(unique_rois, size=n_sampled_rois, replace=False)
        pseudo_dffs_list = []

        for roi in sampled_rois:
            ridx = roi_ids == roi
            roi_keys = roi_keys_all[ridx]
            stim_set = stim_ids[ridx]
            roi_key = roi_keys[0]
            stim_list = np.unique(stim_set)

            for stim in stim_list:
                tidx = np.where((roi_ids == roi) & (stim_ids == stim))[0]
                if len(tidx) == 0:
                    continue

                real_trials = trial_dffs_all[tidx]

                # Fetch baseline DFF trace for this ROI
                baseline_dff = (VM['twophoton'].Rawf2P & roi_key).fetch1('rawf')[0][baseline_window[0]:baseline_window[1]]

                pseudo_trials = analyzeEvoked2P.create_pseudo_trials_from_baseline(
                    baseline_dff=baseline_dff,
                    number_of_trials=len(real_trials),
                    trial_length=min_len,
                    baseline_start_idx=baseline_start_idx,
                    max_offset=max_offset,
                    zscore_to_baseline=zscore_to_baseline,
                    within_trial_baseline_frames=baseline_window_for_z,
                    bootstrap=False
                )

                pseudo_dffs_list.extend(pseudo_trials)

        pseudo_dfs_iteration[f"iter_{i}"] = pd.DataFrame(pseudo_dffs_list)

    return pseudo_dfs_iteration


# %%

# def process_and_filter_response_matrix(
#     dff_trials,
#     roi_ids,
#     stim_ids,
#     roi_keys,
#     kernel_size=15,
#     z_score=False,
#     use_prominence=False,
#     prominence_val=1.96,
#     peak_thresh=1.96,
#     group_threshold=2,
#     proportion_thresh=0.5,
#     std_min=0.1,
#     std_max=30,
#     time_avg_sort='peak_time',
#     time_variance='peak_time',
#     null_thresh=-10,
#     sort=True,
#     frame_range = (105, 300),
#     peak_thresh_mean= 1.96,
#     sampling_interval=0.032958316
# ):
#     """
#     Full pipeline to extract features, group by ROI/stim, filter by reliability, 
#     and return a 2D array of trial-averaged DFFs.

#     Parameters
#     ----------
#     dff_trials : list of np.arrays
#         Trial-wise DFF traces.
#     roi_ids, stim_ids : list-like
#         Same length as dff_trials, indicating ROI and stim identifiers.
#     roi_keys : list
#         List of DataJoint keys to use for SNR or event rate.
#     kernel_size : int
#         Kernel for smoothing during peak detection.
#     z_score : bool
#         Whether to z-score each trace before processing.
#     use_prominence : bool
#         Use prominence instead of height threshold for peak detection.
#     prominence_val : float
#         Minimum prominence or height threshold.
#     peak_thresh : float
#         Peak amplitude threshold for grouping.
#     group_threshold : float
#         Threshold used in group_peaks_by_roi_and_stim.
#     proportion_thresh : float
#         Minimum response proportion for inclusion.
#     std_min, std_max : float
#         Std dev filter range on peak time or COM.
#     sort_by_com : bool
#         Whether to sort output array by average COM or peak time.
#     time_variance : 'peak_time' or 'com'
#         uses std to remove trials that 

#     Returns
#     -------
#     filtered_array : 2D np.ndarray
#         (ROIs × timepoints) array of average DFF traces after filtering and sorting.
#     """
#     # Step 1: Extract metrics from each trial
#     peak_array, auc_array, fwhm_array, peak_time_array, com_array, com_times = analyzeEvoked2P.process_trig_dff_trials(
#         dff_trials,
#         kernel_size=kernel_size,
#         z_score=z_score,
#         use_prominence=use_prominence,
#         prominence_val=prominence_val,
#         peak_thresh=peak_thresh
#     )
    
#     # Step 2
#     # Group by ROI and stim for each feature
#     peak_amp, peak_proportions_dict, proportions = group_peaks_by_roi_and_stim(
#         peak_array, roi_ids, stim_ids, peak_thresh
#     )
    
#     auc, auc_proportions_dict, _ = group_peaks_by_roi_and_stim(
#         auc_array, roi_ids, stim_ids, null_thresh
#     )
    
#     fwhm, fwhm_proportions_dict, _ = group_peaks_by_roi_and_stim(
#         fwhm_array, roi_ids, stim_ids, null_thresh
#     )
    
#     peak_times, peak_times_proportions_dict, _ = group_peaks_by_roi_and_stim(
#         peak_time_array*sampling_interval, roi_ids, stim_ids, null_thresh
#     )
    
#     com, com_proportions_dict, _ = group_peaks_by_roi_and_stim(
#         com_array*sampling_interval, roi_ids, stim_ids, null_thresh
#     )


#     # Extract trials
#     roi_trials, _, _ = group_peaks_by_roi_and_stim(
#         dff_trials, roi_ids, stim_ids, null_thresh
#     )
    
#     averaged_traces_all = average_arrays_within_dict(roi_trials)
    
#     peak_array_mean_trial, auc_array_mean_trial, fwhm_array_mean_trial,peak_time_array_mean_trial, com_array_mean_trial, com_times_mean_trial= analyzeEvoked2P.process_trig_dff_trials(
#         dict_of_arrays_to_2d_array_padded(averaged_traces_all),
#         kernel_size=kernel_size,
#         z_score=z_score,
#         use_prominence=use_prominence,
#         prominence_val=prominence_val,
#         peak_thresh=0
#     )
    
#     roi_keys_dict = group_dicts_by_roi_and_stim(
#         roi_keys, roi_ids, stim_ids
#     )
    
            
#     roi_keys_list = []

#     for (roi_id, stim_id), dicts in roi_keys_dict.items():
#         if len(dicts) == 0:
#             continue  # skip empty
#         first_entry = dict(dicts[0])  # shallow copy of the first dict
#         first_entry['roi_id_extended_dataset'] = roi_id
#         first_entry['stim_id_extended_dataset'] = stim_id
#         roi_keys_list.append(first_entry)
    
#     # Get SNR and event rate from the baseline period, The loop is unfortunately necessary due to how the fetch works
#     snr_values = []
#     snr_keys = []
#     event_rate_values = []

#     for roi in roi_keys_list:
#         # Fetch SNR and associated key (e.g. ROI, session, etc.)
#         snr, key = (VM['twophoton'].Snr2P & roi).fetch('snr', 'KEY')
#         event_rate=(VM['twophoton'].Snr2P & roi).fetch('events_per_min')
        
#         event_rate_values.extend(event_rate)
#         snr_values.extend(snr)
#         snr_keys.extend(key)

    
#     # Step 5: Apply same response filter across all using peak_proportions_dict and proportion_thresh
#     peak_amp_std = get_std_com_above_threshold(peak_amp, peak_proportions_dict, threshold=proportion_thresh)
#     peak_amp_avg = get_avg_com_above_threshold(peak_amp, peak_proportions_dict, threshold=proportion_thresh)
    
#     auc_std = get_std_com_above_threshold(auc, peak_proportions_dict, threshold=proportion_thresh)
#     auc_avg = get_avg_com_above_threshold(auc, peak_proportions_dict, threshold=proportion_thresh)
    
#     fwhm_std = get_std_com_above_threshold(fwhm, peak_proportions_dict, threshold=proportion_thresh)
#     fwhm_avg = get_avg_com_above_threshold(fwhm, peak_proportions_dict, threshold=proportion_thresh)
    
#     peak_time_std = get_std_com_above_threshold(peak_times, peak_proportions_dict, threshold=proportion_thresh)
#     peak_time_avg = get_avg_com_above_threshold(peak_times, peak_proportions_dict, threshold=proportion_thresh)
    
#     com_std = get_std_com_above_threshold(com, peak_proportions_dict, threshold=proportion_thresh)
#     com_avg = get_avg_com_above_threshold(com, peak_proportions_dict, threshold=proportion_thresh)

    
#     if time_variance=='peak_time':
#         feature_std_filter=peak_time_std
#     else:
#         feature_std_filter=com_std
        
#     if time_avg_sort=='peak_time':
#         feature_avg =peak_time_avg
#     else:
#         feature_avg =com_avg
    
        
#     response_proportions = filter_dict_by_threshold(peak_proportions_dict, feature_std_filter, std_min, std_max)
    
#     filtered_trials = filter_dict_by_threshold(roi_trials, feature_std_filter, std_min, std_max)
#     filtered_avg_metric = filter_dict_by_threshold(feature_avg, feature_std_filter, std_min, std_max)
    

#     # # Step 4: Trial averaging
#     averaged_traces = average_arrays_within_dict(filtered_trials)

#     # Step 5: Optional sorting
#     if sort:
#         sorted_keys = sorted(filtered_avg_metric, key=filtered_avg_metric.get)
#         averaged_traces = {k: averaged_traces[k] for k in sorted_keys}
    
    
#     # Construct dataframe rows by merging results into dicts
#     result_records = []
    
#     for entry in roi_keys_list:
#         key = (entry['roi_id_extended_dataset'], entry['stim_id'])
    
#         record = {
#             **entry,  # contains roi_id, stim_id, and any DataJoint metadata
#             # 'peak_array_mean_trial':peak_array_mean_trial,
#             'peak_amp_avg': peak_amp_avg.get(key, np.nan),
#             'peak_amp_std': peak_amp_std.get(key, np.nan),
#             'auc_avg': auc_avg.get(key, np.nan),
#             'auc_std': auc_std.get(key, np.nan),
#             'fwhm_avg': fwhm_avg.get(key, np.nan),
#             'fwhm_std': fwhm_std.get(key, np.nan),
#             'peak_time_avg': peak_time_avg.get(key, np.nan),
#             'peak_time_std': peak_time_std.get(key, np.nan),
#             'com_avg': com_avg.get(key, np.nan),
#             'com_std': com_std.get(key, np.nan),
#             'response_proportion': peak_proportions_dict.get(key, np.nan),
#             'trials_arrays':roi_trials.get(key),
#             'averaged_traces_all':averaged_traces_all.get(key, np.nan),
#             'roi_keys':roi_keys_dict.get(key, np.nan)
#         }
    
#         # # You could also attach filtered avg trace or peak array mean for plotting
#         # if key in averaged_traces:
#         #     record['trace_avg'] = averaged_traces[key]
#         # else:
#         #     record['trace_avg'] = np.nan
    
#         result_records.append(record)

#     # Create dataframe
#     results_df = pd.DataFrame(result_records)
#     results_df['peak_array_mean_trial'] = peak_array_mean_trial
#     results_df['snr'] = snr_values
#     results_df['event_rate'] = event_rate_values
#     results_df['com_array_mean_trial'] = com_array_mean_trial*sampling_interval
#     results_df['peak_time_array_mean_trial'] = peak_time_array_mean_trial*sampling_interval
#     results_df['peak_time_calc_abs_diff_mean_trial']=np.abs((peak_time_array_mean_trial*sampling_interval)-results_df['peak_time_avg'].to_numpy())
#     results_df['com_calc_abs_diff_mean_trial']=np.abs((com_array_mean_trial*sampling_interval)-results_df['com_avg'].to_numpy())
    

#     # # Step 6: Convert to 2D array (ROIs × time)
#     return dict_of_arrays_to_2d_array_padded(averaged_traces), roi_keys_list,results_df

# %%

def process_and_filter_response_matrix(
    dff_trials,
    roi_ids,
    stim_ids,
    roi_keys,
    time_axes,
    kernel_size=15,
    z_score=False,
    use_prominence=False,
    prominence_val=1.96,
    peak_thresh=1.96,
    group_threshold=2,
    proportion_thresh=0.5,
    std_min=0.1,
    std_max=30,
    time_avg_sort='peak_time',
    time_variance='peak_time',
    null_thresh=-10,
    sort=True,
    frame_range=(105, 300),
    peak_thresh_mean=1.96,
    sampling_interval=0.032958316,
    subsample=True,
    n_subsample=200,
    random_seed=None,
    scramble_stim_ids=False,
    max_trials_per_group=10,
    min_width_val=2
):
    
    """
    Full pipeline to extract features, group by ROI/stim, filter by reliability, 
    and return a 2D array of trial-averaged DFFs.

    Parameters
    ----------
    dff_trials : list of np.arrays
        Trial-wise DFF traces.
    roi_ids, stim_ids : list-like
        Same length as dff_trials, indicating ROI and stim identifiers.
    roi_keys : list
        List of DataJoint keys to use for SNR or event rate.
    kernel_size : int
        Kernel for smoothing during peak detection.
    z_score : bool
        Whether to z-score each trace before processing.
    use_prominence : bool
        Use prominence instead of height threshold for peak detection.
    prominence_val : float
        Minimum prominence or height threshold.
    peak_thresh : float
        Peak amplitude threshold for grouping.
    group_threshold : float
        Threshold used in group_peaks_by_roi_and_stim.
    proportion_thresh : float
        Minimum response proportion for inclusion.
    std_min, std_max : float
        Std dev filter range on peak time or COM.
    sort_by_com : bool
        Whether to sort output array by average COM or peak time.
    time_variance : 'peak_time' or 'com'
        uses std to remove trials that 

    Returns
    -------
    filtered_array : 2D np.ndarray
        (ROIs × timepoints) array of average DFF traces after filtering and sorting.
    """
    
    import numpy as np
    import pandas as pd
    import random
    from schemas import spont_timescales
    from schemas import twop_opto_analysis

    
    if random_seed is not None:
        np.random.seed(random_seed)
        # random.seed(random_seed)
    
    if subsample:
        unique_roi_stim = list(set(zip(roi_ids, stim_ids)))
        if random_seed is not None:
            random.seed(random_seed)
        sampled_roi_stim = random.sample(unique_roi_stim, min(n_subsample, len(unique_roi_stim)))
        mask = [i for i, (r, s) in enumerate(zip(roi_ids, stim_ids)) if (r, s) in sampled_roi_stim]
        dff_trials = [dff_trials[i] for i in mask]
        roi_ids = [roi_ids[i] for i in mask]
        stim_ids = [stim_ids[i] for i in mask]
        roi_keys = [roi_keys[i] for i in mask]
        time_axes = [time_axes[i] for i in mask]

    if scramble_stim_ids:
        # if random_seed is not None:
        #     np.random.seed(random_seed)

        # Scramble the order of dff_trials only
        dff_trials = [dff_trials[i] for i in np.random.permutation(len(dff_trials))]
        # Ensure all dff_trials are the same length — trim to shortest
        min_len = min(len(trace) for trace in dff_trials)
        dff_trials = [trace[:min_len] for trace in dff_trials]
        time_axes = [trace[:min_len] for trace in time_axes]


        # t_axes_all = np.array(trial_data['time_axis_sec'], dtype=object)

    # Step 1: Extract metrics from each trial
    peak_array, auc_array, fwhm_array, peak_time_array, com_array, com_times = analyzeEvoked2P.process_trig_dff_trials(
        dff_trials,
        kernel_size=kernel_size,
        z_score=z_score,
        use_prominence=use_prominence,
        prominence_val=prominence_val,
        peak_thresh=peak_thresh,
        min_width_val=min_width_val
    )
    
    # Step 2
    # Group by ROI and stim for each feature
    peak_amp, peak_proportions_dict, proportions = group_peaks_by_roi_and_stim(
        peak_array, roi_ids, stim_ids, group_threshold,max_trials_per_group
    )
    
    auc, auc_proportions_dict, _ = group_peaks_by_roi_and_stim(
        auc_array, roi_ids, stim_ids, null_thresh,max_trials_per_group
    )
    
    fwhm, fwhm_proportions_dict, _ = group_peaks_by_roi_and_stim(
        fwhm_array, roi_ids, stim_ids, null_thresh
    )
    
    peak_times, peak_times_proportions_dict, _ = group_peaks_by_roi_and_stim(
        peak_time_array*sampling_interval, roi_ids, stim_ids, null_thresh,max_trials_per_group
    )
    
    com, com_proportions_dict, _ = group_peaks_by_roi_and_stim(
        com_array*sampling_interval, roi_ids, stim_ids, null_thresh,max_trials_per_group
    )


    # Extract trials
    roi_trials, _, _ = group_peaks_by_roi_and_stim(
        dff_trials, roi_ids, stim_ids, null_thresh,max_trials_per_group
    )
    
    # Extract trial time_axes
    roi_time_axes, _, _ = group_peaks_by_roi_and_stim(
        time_axes, roi_ids, stim_ids, null_thresh,max_trials_per_group
    )
    
    averaged_traces_all = average_arrays_within_dict(roi_trials)
    
    peak_array_mean_trial, auc_array_mean_trial, fwhm_array_mean_trial,peak_time_array_mean_trial, com_array_mean_trial, com_times_mean_trial= analyzeEvoked2P.process_trig_dff_trials(
        dict_of_arrays_to_2d_array_padded(averaged_traces_all),
        kernel_size=kernel_size,
        z_score=z_score,
        use_prominence=use_prominence,
        prominence_val=prominence_val,
        peak_thresh=0,
        min_width_val=min_width_val
    )
    
    roi_keys_dict = group_dicts_by_roi_and_stim(
        roi_keys, roi_ids, stim_ids,max_trials_per_group
    )
    
            
    roi_keys_list = []

    for (roi_id, stim_id), dicts in roi_keys_dict.items():
        if len(dicts) == 0:
            continue  # skip empty
        first_entry = dict(dicts[0])  # shallow copy of the first dict
        first_entry['roi_id_extended_dataset'] = roi_id
        first_entry['stim_id_extended_dataset'] = stim_id
        roi_keys_list.append(first_entry)
    

    
    # Step 5: Apply same response filter across all using peak_proportions_dict and proportion_thresh
    peak_amp_std = get_std_com_above_threshold(peak_amp, peak_proportions_dict, threshold=proportion_thresh)
    peak_amp_avg = get_avg_com_above_threshold(peak_amp, peak_proportions_dict, threshold=proportion_thresh)
    
    auc_std = get_std_com_above_threshold(auc, peak_proportions_dict, threshold=proportion_thresh)
    auc_avg = get_avg_com_above_threshold(auc, peak_proportions_dict, threshold=proportion_thresh)
    
    fwhm_std = get_std_com_above_threshold(fwhm, peak_proportions_dict, threshold=proportion_thresh)
    fwhm_avg = get_avg_com_above_threshold(fwhm, peak_proportions_dict, threshold=proportion_thresh)
    
    peak_time_std = get_std_com_above_threshold(peak_times, peak_proportions_dict, threshold=proportion_thresh)
    peak_time_avg = get_avg_com_above_threshold(peak_times, peak_proportions_dict, threshold=proportion_thresh)
    
    com_std = get_std_com_above_threshold(com, peak_proportions_dict, threshold=proportion_thresh)
    com_avg = get_avg_com_above_threshold(com, peak_proportions_dict, threshold=proportion_thresh)

    
    if time_variance=='peak_time':
        feature_std_filter=peak_time_std
    else:
        feature_std_filter=com_std
        
    if time_avg_sort=='peak_time':
        feature_avg =peak_time_avg
    else:
        feature_avg =com_avg
    
        
    response_proportions = filter_dict_by_threshold(peak_proportions_dict, feature_std_filter, std_min, std_max)
    
    filtered_trials = filter_dict_by_threshold(roi_trials, feature_std_filter, std_min, std_max)
    filtered_avg_metric = filter_dict_by_threshold(feature_avg, feature_std_filter, std_min, std_max)
    

    # # Step 4: Trial averaging
    averaged_traces = average_arrays_within_dict(filtered_trials)

    # Step 5: Optional sorting
    if sort:
        sorted_keys = sorted(filtered_avg_metric, key=filtered_avg_metric.get)
        averaged_traces = {k: averaged_traces[k] for k in sorted_keys}
    
    
    # Construct dataframe rows by merging results into dicts
    result_records = []
    
    for entry in roi_keys_list:
        key = (entry['roi_id_extended_dataset'], entry['stim_id'])
    
        record = {
            **entry,  # contains roi_id, stim_id, and any DataJoint metadata
            'peak_amp_avg': peak_amp_avg.get(key, np.nan),
            'peak_amp_std': peak_amp_std.get(key, np.nan),
            'auc_avg': auc_avg.get(key, np.nan),
            'auc_std': auc_std.get(key, np.nan),
            'fwhm_avg': fwhm_avg.get(key, np.nan),
            'fwhm_std': fwhm_std.get(key, np.nan),
            'peak_time_avg': peak_time_avg.get(key, np.nan),
            'peak_time_std': peak_time_std.get(key, np.nan),
            'com_avg': com_avg.get(key, np.nan),
            'com_std': com_std.get(key, np.nan),
            'response_proportion': peak_proportions_dict.get(key, np.nan),
            'trials_arrays':roi_trials.get(key),
            'time_axes':roi_time_axes.get(key),
            'averaged_traces_all':averaged_traces_all.get(key, np.nan),
            'roi_keys':roi_keys_dict.get(key, np.nan),
            'peak_time_by_trial':peak_times.get(key, np.nan),
            'com_by_trial':com.get(key, np.nan),
            'peak_amp_by_trial':peak_amp.get(key, np.nan),
            'fwhm_by_trial':fwhm.get(key, np.nan),
            'auc_by_trial':auc.get(key, np.nan)
        }
  
    
        result_records.append(record)

    # Create dataframe
    results_df = pd.DataFrame(result_records)
    results_df['peak_array_mean_trial'] = peak_array_mean_trial


    # results_df['event_rate'] = event_rate_values
    results_df['com_array_mean_trial'] = com_array_mean_trial*sampling_interval
    results_df['peak_time_array_mean_trial'] = peak_time_array_mean_trial*sampling_interval
    results_df['peak_time_calc_abs_diff_mean_trial']=np.abs((peak_time_array_mean_trial*sampling_interval)-results_df['peak_time_avg'].to_numpy())
    results_df['com_calc_abs_diff_mean_trial']=np.abs((com_array_mean_trial*sampling_interval)-results_df['com_avg'].to_numpy())
    

    # # Step 6: Convert to 2D array (ROIs × time)
    return peak_array, roi_keys_list,results_df


# %%

def process_and_filter_response_matrix_from_df(
    entire_df,
    kernel_size=15,
    z_score=False,
    use_prominence=False,
    prominence_val=1.96,
    peak_thresh=1.96,
    group_threshold=2,
    proportion_thresh=0.5,
    std_min=0.1,
    std_max=30,
    time_avg_sort='peak_time',
    time_variance='peak_time',
    null_thresh=-10,
    sort=True,
    frame_range=(105, 300),
    peak_thresh_mean=1.96,
    sampling_interval=0.032958316,
    subsample=True,
    n_subsample=200,
    random_seed=None,
    scramble_stim_ids=False,
    max_trials_per_group=10,
    min_width_val=10
):
    
    """
    Full pipeline to extract features, group by ROI/stim, filter by reliability, 
    and return a 2D array of trial-averaged DFFs.

    Parameters
    ----------
    dff_trials : list of np.arrays
        Trial-wise DFF traces.
    roi_ids, stim_ids : list-like
        Same length as dff_trials, indicating ROI and stim identifiers.
    roi_keys : list
        List of DataJoint keys to use for SNR or event rate.
    kernel_size : int
        Kernel for smoothing during peak detection.
    z_score : bool
        Whether to z-score each trace before processing.
    use_prominence : bool
        Use prominence instead of height threshold for peak detection.
    prominence_val : float
        Minimum prominence or height threshold.
    peak_thresh : float
        Peak amplitude threshold for grouping.
    group_threshold : float
        Threshold used in group_peaks_by_roi_and_stim.
    proportion_thresh : float
        Minimum response proportion for inclusion.
    std_min, std_max : float
        Std dev filter range on peak time or COM.
    sort_by_com : bool
        Whether to sort output array by average COM or peak time.
    time_variance : 'peak_time' or 'com'
        uses std to remove trials that 

    Returns
    -------
    filtered_array : 2D np.ndarray
        (ROIs × timepoints) array of average DFF traces after filtering and sorting.
    """
    
    import numpy as np
    import pandas as pd
    import random
    
    entire_df = entire_df.rename(
        columns={
            'roi_id': 'roi_id_extended_dataset',
            'stim_id': 'stim_id_extended_dataset'
        }
    )


    # This portion of the code does single trial anlysis and groups by combination of roi and sitm id


    # Run your trial analysis
    dff_trials = entire_df['trials_arrays'].to_list()

    peak_array, auc_array, fwhm_array, peak_time_array, com_array, com_times = analyzeEvoked2P.process_trig_dff_trials(
        dff_trials,
        kernel_size=kernel_size,
        z_score=z_score,
        use_prominence=use_prominence,
        prominence_val=prominence_val,
        peak_thresh=peak_thresh,
        min_width_val=min_width_val
    )

    # Add the new metrics as columns to your original DataFrame
    entire_df['peak_amp_by_trial'] = peak_array
    entire_df['auc_by_trial'] = auc_array
    entire_df['fwhm_by_trial'] = fwhm_array
    entire_df['peak_time_by_trial'] = peak_time_array*sampling_interval
    entire_df['com_by_trial'] = com_array*sampling_interval
    entire_df['com_time_by_trial'] = com_times*sampling_interval

    # List of metrics to group
    metrics = ['peak_amp_by_trial', 'auc_by_trial', 'fwhm_by_trial', 'peak_time_by_trial', 'com_by_trial', 'com_time_by_trial']

    # Group by roi_id and stim_id, aggregating each metric as a list
    grouped_df = entire_df.groupby(['roi_id_extended_dataset', 'stim_id_extended_dataset'])[metrics].agg(list).reset_index()

    grouped_df['response_proportion'] = grouped_df['peak_amp_by_trial'].apply(
        lambda peak_list: np.sum(~np.isnan(peak_list)) / len(peak_list) if len(peak_list) > 0 else np.nan
    )

    for metric in metrics:
        base_name = metric.replace('_by_trial', '')
        grouped_df[f'{base_name}_avg'] = grouped_df[metric].apply(lambda x: np.nanmean(x) if len(x) > 0 else np.nan)
        grouped_df[f'{base_name}_std'] = grouped_df[metric].apply(lambda x: np.nanstd(x) if len(x) > 0 else np.nan)

    # This portion of the code expands roi_keys to be saved in final dataframe

    # Extract the source DataFrame
    df = entire_df.copy()

    # Extract roi_keys (dict or tuple) and add as column if not already
    df['roi_keys'] = df['roi_key']

    # Drop rows where roi_keys are missing
    df_filtered = df.dropna(subset=['roi_keys'])

    # Group by roi_id and stim_id and take the first roi_key in each group
    # Group by roi_id and stim_id, keep the first roi_keys dict per group
    grouped_df_keys = (
        df.groupby(['roi_id_extended_dataset', 'stim_id_extended_dataset'])['roi_keys']
        .first()
        .reset_index()
    )

    # Expand each roi_keys dict into separate columns
    roi_keys_expanded = grouped_df_keys['roi_keys'].apply(pd.Series)

    # Concatenate roi_id, stim_id, and the expanded roi_keys
    # Concatenate roi_id, stim_id, original roi_keys, and the expanded fields
    final_df_keys = pd.concat([
        grouped_df_keys[['roi_id_extended_dataset', 'stim_id_extended_dataset', 'roi_keys']], 
        roi_keys_expanded
    ], axis=1)


    # This portion of the code calculates the response properties of the mean of trials across conditions

    # Function to stack the pseudo_trial arrays for each group
    def stack_trials(trials):
        return np.stack(trials.to_list(), axis=0)  # shape: (n_trials, timepoints)

    # Group by roi_id and stim_id, and stack the trials
    trial_grouped_df = (
        entire_df
        .groupby(['roi_id_extended_dataset', 'stim_id_extended_dataset'])['trials_arrays']
        .apply(stack_trials)
        .reset_index()
    )

    trial_grouped_df['averaged_traces_all'] = trial_grouped_df['trials_arrays'].apply(lambda x: np.mean(x, axis=0))


    peak_array_mean_trial, auc_array_mean_trial, fwhm_array_mean_trial,peak_time_array_mean_trial, com_array_mean_trial, com_times_mean_trial= analyzeEvoked2P.process_trig_dff_trials(
        trial_grouped_df['averaged_traces_all'],
        kernel_size=kernel_size,
        z_score=z_score,
        use_prominence=use_prominence,
        prominence_val=prominence_val,
        peak_thresh=0,
        min_width_val=min_width_val
    )

    # Add the new metrics as columns to your original DataFrame
    trial_grouped_df['peak_array_mean_trial'] = peak_array_mean_trial
    trial_grouped_df['auc_mean_trial'] = auc_array_mean_trial
    trial_grouped_df['fwhm_mean_trial'] = fwhm_array_mean_trial
    trial_grouped_df['peak_time_array_mean_trial'] = peak_time_array_mean_trial
    trial_grouped_df['com_array_mean_trial'] = com_array_mean_trial
    trial_grouped_df['com_time_mean_trial'] = com_times_mean_trial



    # Start with the first merge
    results_df = pd.merge(final_df_keys,grouped_df, on=['roi_id_extended_dataset', 'stim_id_extended_dataset'], how='inner')

    # Then merge with the keys dataframe
    results_df = pd.merge(results_df, trial_grouped_df, on=['roi_id_extended_dataset', 'stim_id_extended_dataset'], how='inner')
    
    
    return results_df

# %%

def filter_and_summarize(df, original_df, steps, features_to_track, label="filtered"):
    """
    Apply a list of filter steps to `df`, log summary after each,
    and return the filtered df and summary log.
    
    Parameters:
    - df: The DataFrame to filter (copy will be made)
    - original_df: The full dataset to compute 'roi_occurrence_all' from
    - steps: List of tuples (description, filter_function), applied sequentially
    - features_to_track: List of feature names to include in nanmean summaries
    - label: Optional string label for the output summary DataFrame name
    
    Returns:
    - filtered_df: the final filtered DataFrame
    - summary_df: a DataFrame logging filtering steps
    """
    import numpy as np
    import pandas as pd

    def summarize_step(df, step_name, feature_list):
        summary = {
            'step': step_name,
            'n_rows': len(df),
        }
        # Mean occurrence count from the original (not current filtered) dataset
        summary['roi_occurrence_all'] = np.nanmean(
            df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts())
        )
        for feat in feature_list:
            summary[feat + '_nanmean'] = np.nanmean(df[feat]) if feat in df.columns else np.nan
        return pd.DataFrame([summary])
    
    # Initialize
    filt = df.copy()
    summary_df = summarize_step(filt, 'original', features_to_track)

    for desc, filter_func in steps:
        # Apply filter
        filt = filter_func(filt)
        # Recompute roi_occurrence_all using original df (full context)
        filt['roi_occurrence_all'] = filt['roi_id_extended_dataset'].map(
            filt['roi_id_extended_dataset'].value_counts()
        )
        # Log this step
        summary_df = pd.concat([summary_df, summarize_step(filt, desc, features_to_track)], ignore_index=True)

    return filt, summary_df

