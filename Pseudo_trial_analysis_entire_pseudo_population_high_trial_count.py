# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:47:34 2025

@author: jec822
"""




# %%

# %%


pseudo_dfs_iteration_M2_high=calculate_single_trial_features.generate_pseudo_trials(
    single_trial_data,
    n_iters=1,
    baseline_window=(1000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=7000,
    zscore_to_baseline=True,
    baseline_window_for_z=80
)

# joblib.dump(pseudo_dfs_iteration_M2, 'pseudo_dfs_iteration_M2_20_it.joblib')


# # %%
# import joblib
# single_trial_data_standard_non_stimd_M2_all_cells = joblib.load('single_trial_data_standard_non_stimd_M2_all_cells.joblib')

# single_trial_data_standard_non_stimd_V1_all_cells = joblib.load('single_trial_data_standard_non_stimd_V1_all_cells.joblib')

# pseudo_dfs_iteration_V1=joblib.load('pseudo_dfs_iteration_V1_20_it.joblib')
# pseudo_dfs_iteration_M2=joblib.load('pseudo_dfs_iteration_M2_20_it.joblib')

# # %%

# pseudo_trials_dffs_iter_V1=pseudo_dfs_iteration_V1['iter_0']
# pseudo_trials_list_V1 = [row.values for _, row in pseudo_trials_dffs_iter_V1.iterrows()]

pseudo_trials_dffs_iter_M2=pseudo_dfs_iteration_M2['iter_0']
pseudo_trials_list_M2 = [row.values for _, row in pseudo_trials_dffs_iter_M2.iterrows()]


pseudo_trial_data=pseudo_trials_list_M2

filtered_array_M2_pseudo,a,result_M2_pseudo_high= calculate_single_trial_features.process_and_filter_response_matrix(
    dff_trials=pseudo_trial_data,
    roi_ids=single_trial_data['roi_ids'],
    stim_ids=single_trial_data['stim_ids'],
    roi_keys=single_trial_data['roi_keys'],
    kernel_size=9,
    peak_thresh=1.96,
    group_threshold=2,
    time_variance='peak_time', # or False to use COM
    subsample=False,
    max_trials_per_group=20
)
# %%

single_trial_data=result_M2_high_trial_count_non_stimd

filtered_array_M2,a,result_M2_high = calculate_single_trial_features.process_and_filter_response_matrix(
    dff_trials=single_trial_data['trials_arrays'],
    roi_ids=single_trial_data['roi_id'],
    stim_ids=single_trial_data['stim_id'],
    roi_keys=single_trial_data['roi_keys'],
    time_axes=single_trial_data['time_axis_sec'],
    kernel_size=7,
    peak_thresh=2.94,
    group_threshold=2,
    time_variance='peak_time',
    subsample=False,
    scramble_stim_ids=False,
    max_trials_per_group=5
)
# need to look into alignment of trials within this function


# %%

# ---- Common Setup ----
features_to_track = [
    'peak_time_std', 'com_std', 'auc_avg', 'peak_array_mean_trial', 'peak_amp_avg',
    'peak_time_array_mean_trial', 'com_array_mean_trial', 'response_proportion',
    'com_calc_abs_diff_mean_trial', 'peak_time_calc_abs_diff_mean_trial'
]

# ---- Define filters ----
filter_steps = [
    # ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 2]),
    ('response_proportion >= 0.6', lambda df: df[df['response_proportion'] >= 0.6]),
    # ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 5]),
    # ('peak_array_mean_trial > 0', lambda df: df[df['peak_array_mean_trial'] > 2]),
    ('peak_time_std <= 0.5', lambda df: df[df['peak_time_std'] <= 0.5]),
    # ('peak_time_avg <= 0.5', lambda df: df[df['peak_time_avg'] >= 1.0]),
    # ('peak_time_calc_abs_diff_mean_trial <= 0.4', lambda df: df[df['peak_time_calc_abs_diff_mean_trial'] <= 0.2]),
    # ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 2]),
    # ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 0]),
]
# %%
# ---- Process real ----
result_M2_high['is_shuffled'] = False
result_M2_high_filt, summaries_M2 = calculate_single_trial_features.filter_and_summarize(result_M2_high, result_M2_high, filter_steps, features_to_track, label="real")


# ---- Process pseudo ----
# result_M2_pseudo['is_shuffled'] = True
# result_M2_pseudo_high_filt, summaries_M2_pseudo = calculate_single_trial_features.filter_and_summarize(result_M2_pseudo_high, result_M2_pseudo_high, filter_steps, features_to_track, label="pseudo")


# %%

calculate_single_trial_features.prepare_and_plot_heatmap(result_M2_high_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-1,
    vmax=1,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None)

calculate_single_trial_features.prepare_and_plot_heatmap(result_M2_pseudo_high_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-1,
    vmax=1,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None)

