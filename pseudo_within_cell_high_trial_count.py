# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:42:45 2025

@author: jec822
"""


trial_data=result_M2_high_trial_count_non_stimd_filt

pseudo_dfs_iteration_M2=calculate_single_trial_features.generate_pseudo_trials_select_cells(
    roi_ids = np.array(trial_data['roi_id_extended_dataset']),
    stim_ids = np.array(trial_data['stim_id']),
    roi_keys_all = np.array(trial_data['roi_keys']),
    trials = trial_data['trials_arrays'],
    n_iters=1,
    baseline_window=(1000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80,
    repeat_count=1 #only one stim type
)
# %%

pseudo_results_multi_iter_M2 = {}

# Loop over all iterations in your pseudo dataframe dictionary
for iter_label, pseudo_trials_dffs_iter in pseudo_dfs_iteration_M2.items():

    # Run your processing function
    result_M2_pseudo= calculate_single_trial_features.process_and_filter_response_matrix_from_df(
        entire_df=pseudo_trials_dffs_iter,
        kernel_size=7,
        peak_thresh=2.96,
        group_threshold=2,
        time_variance='peak_time', # or False to use COM
        subsample=False,
        min_width_val=2
    )
    
    # Store the result DataFrame in the dictionary
    pseudo_results_multi_iter_M2[iter_label] =  result_M2_pseudo
    
    
# %% Features to filter responsive cells

# ---- Common Setup ----
features_to_track = [
    'peak_time_std', 'com_std', 'auc_avg', 'peak_array_mean_trial', 'peak_amp_avg',
    'peak_time_array_mean_trial', 'com_array_mean_trial', 'response_proportion',
    'com_calc_abs_diff_mean_trial', 'peak_time_calc_abs_diff_mean_trial'
]

# ---- Define filters ----
filter_steps = [
    ('response_proportion >= 0.6', lambda df: df[df['response_proportion'] >= 0.6]),
    # ('peak_time_std <= 0.5', lambda df: df[df['peak_time_std'] <= 0.75]),
    # ('peak_time_avg <= 0.5', lambda df: df[df['peak_time_avg'] >= 1.0]),
    ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 2]),
    ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 0])
    ]
# %%

pseudo_results_filt_multi_iter_M2 = {}

# Loop over all iterations in your pseudo dataframe dictionary
for iter_label, pseudo_trials_dffs_iter in pseudo_results_multi_iter_M2.items():

    # Run your processing function
    result_M2_pseudo_filt, summaries_M2_pseudo = calculate_single_trial_features.filter_and_summarize(pseudo_trials_dffs_iter, pseudo_trials_dffs_iter, filter_steps, features_to_track, label="pseudo")

    
    # Store the result DataFrame in the dictionary
    pseudo_results_filt_multi_iter_M2[iter_label] =  result_M2_pseudo_filt



