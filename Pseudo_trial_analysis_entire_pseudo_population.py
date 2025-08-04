# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:47:34 2025

@author: jec822
"""

# %%
 #  this didnt make a meaningfull difference and was intensely slow
 
# pseudo_dfs_iteration_M2_sub_sampled=calculate_single_trial_features.generate_pseudo_trials_subsampled(
#     single_trial_data_standard_non_stimd_M2_all_cells,
#     n_iters=1,
#     baseline_window=(5000, 10000),
#     trial_crop_idx=None,
#     baseline_start_idx=0,
#     max_offset=5000,
#     zscore_to_baseline=True,
#     baseline_window_for_z=80,
#     n_sampled_rois=10000
# )

# pseudo_trials_dffs_iter_M2=pseudo_dfs_iteration_M2_sub_sampled['iter_0']
# pseudo_trials_list_M2 = [row.values for _, row in pseudo_trials_dffs_iter_M2.iterrows()]


# %%

pseudo_dfs_iteration_V1=calculate_single_trial_features.generate_pseudo_trials(
    single_trial_data__V1_standard_non_stimd_sig_False,
    n_iters=1,
    baseline_window=(1000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80
)

joblib.dump(pseudo_dfs_iteration_V1, 'pseudo_dfs_iteration_V1_1_it.joblib')


pseudo_dfs_iteration_M2=calculate_single_trial_features.generate_pseudo_trials(
    single_trial_data__M2_standard_non_stimd_sig_False,
    n_iters=1,
    baseline_window=(1000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80
)

joblib.dump(pseudo_dfs_iteration_M2, 'pseudo_dfs_iteration_M2_1_it.joblib')


# %%
import joblib

pseudo_dfs_iteration_V1=joblib.load('pseudo_dfs_iteration_V1_1_it.joblib')
pseudo_dfs_iteration_M2=joblib.load('pseudo_dfs_iteration_M2_1_it.joblib')

# %%

pseudo_trials_dffs_iter_V1=pseudo_dfs_iteration_V1['iter_0']
pseudo_trials_list_V1 = [row.values for _, row in pseudo_trials_dffs_iter_V1.iterrows()]

pseudo_trials_dffs_iter_M2=pseudo_dfs_iteration_M2['iter_0']
pseudo_trials_list_M2 = [row.values for _, row in pseudo_trials_dffs_iter_M2.iterrows()]

# %% this repeats over the 20 repeats of the entire pseudo population

# Dictionary to hold result_dfs for each iteration
# pseudo_results_mult_iter_V1 = {}

# # Loop over all iterations in your pseudo dataframe dictionary
# for iter_label, pseudo_trials_dffs_iter_V1 in pseudo_dfs_iteration_V1.items():
#     # Convert each row of the DataFrame into a list of dF/F values (1 per ROI)
#     pseudo_trials_list_V1 = [row.values for _, row in pseudo_trials_dffs_iter_V1.iterrows()]

#     # Run your processing function
#     filtered_array_V1_pseudo, a, result_V1_pseudo = calculate_single_trial_features.process_and_filter_response_matrix(
#         dff_trials=pseudo_trials_list_V1,
#         roi_ids=single_trial_data_standard_non_stimd_V1_all_cells['roi_ids'],
#         stim_ids=single_trial_data_standard_non_stimd_V1_all_cells['stim_ids'],
#         roi_keys=single_trial_data_standard_non_stimd_V1_all_cells['roi_keys'],
#         kernel_size=15,
#         peak_thresh=1.96,
#         group_threshold=2,
#         time_variance='peak_time'
#     )
    
#     # Store the result DataFrame in the dictionary
#     pseudo_results_mult_iter_V1[iter_label] = result_V1_pseudo

# %%

pseudo_trial_data=pseudo_trials_list_V1
single_trial_data=single_trial_data__V1_standard_non_stimd_sig_False

filtered_array_V1_pseudo,a,result_V1_pseudo= calculate_single_trial_features.process_and_filter_response_matrix(
    dff_trials=pseudo_trial_data,
    roi_ids=single_trial_data['roi_ids'],
    stim_ids=single_trial_data['stim_ids'],
    roi_keys=single_trial_data['roi_keys'],
    time_axes=single_trial_data['time_axis_sec'],
    kernel_size=7,
    peak_thresh=2.96,
    group_threshold=2,
    time_variance='peak_time',  # or False to use COM
    subsample=False,
    min_width_val=2
)


pseudo_trial_data=pseudo_trials_list_M2
single_trial_data=single_trial_data__M2_standard_non_stimd_sig_False

filtered_array_M2_pseudo,a,result_M2_pseudo= calculate_single_trial_features.process_and_filter_response_matrix(
    dff_trials=pseudo_trial_data,
    roi_ids=single_trial_data['roi_ids'],
    stim_ids=single_trial_data['stim_ids'],
    roi_keys=single_trial_data['roi_keys'],
    time_axes=single_trial_data['time_axis_sec'],
    kernel_size=7,
    peak_thresh=2.96,
    group_threshold=2,
    time_variance='peak_time', # or False to use COM
    subsample=False,
    min_width_val=2
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
    ('peak_time_std <= 0.5', lambda df: df[df['peak_time_std'] <= 0.75]),
    # ('peak_time_avg <= 0.5', lambda df: df[df['peak_time_avg'] >= 1.0]),
    # ('peak_time_calc_abs_diff_mean_trial <= 0.4', lambda df: df[df['peak_time_calc_abs_diff_mean_trial'] <= 0.2]),
    ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 2]),
    ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 0]),
]
# %%

# ---- Process real ----
# result_M2['is_shuffled'] = False
# result_M2_filt, summaries_M2 = calculate_single_trial_features.filter_and_summarize(result_M2, result_M2, filter_steps, features_to_track, label="real")

# # ---- Process pseudo ----
# result_M2_pseudo['is_shuffled'] = True
# result_M2_pseudo_filt, summaries_M2_pseudo = calculate_single_trial_features.filter_and_summarize(result_M2_pseudo, result_M2_pseudo, filter_steps, features_to_track, label="pseudo")


# result_V1['is_shuffled'] = False
# result_V1_filt, summaries_V1 = calculate_single_trial_features.filter_and_summarize(result_V1, result_V1, filter_steps, features_to_track, label="real")

# # ---- Process pseudo ----
result_V1_pseudo['is_shuffled'] = True
result_V1_pseudo_filt, summaries_V1_pseudo = calculate_single_trial_features.filter_and_summarize(result_V1_pseudo, result_V1_pseudo, filter_steps, features_to_track, label="pseudo")

# %%

calculate_single_trial_features.prepare_and_plot_heatmap(result_V1_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-2,
    vmax=2,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None
    )

calculate_single_trial_features.prepare_and_plot_heatmap(result_V1_pseudo_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-2,
    vmax=2,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None)

calculate_single_trial_features.prepare_and_plot_heatmap(filtered_result_M2_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-2,
    vmax=2,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None)

calculate_single_trial_features.prepare_and_plot_heatmap(result_M2_pseudo_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-2,
    vmax=2,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None)

# %%


plt.figure()
plt.scatter(non_overlapping_df_sorted['peak_time_avg'].to_numpy(),non_overlapping_df_sorted['roi_occurrence_all'].to_numpy())

plt.figure()
plt.scatter(non_overlapping_df_sorted['peak_time_avg'].to_numpy(),non_overlapping_df_sorted['peak_time_std'].to_numpy())


