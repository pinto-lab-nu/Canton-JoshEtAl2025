# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:27:44 2025

@author: jec822
"""
single_trial_data=single_trial_data_standard_non_stimd_M2_all_cells

filtered_array_M2,a,result_M2_scrambled = calculate_single_trial_features.process_and_filter_response_matrix(
    dff_trials=single_trial_data['trig_dff_trials'],
    roi_ids=single_trial_data['roi_ids'],
    stim_ids=single_trial_data['stim_ids'],
    roi_keys=single_trial_data['roi_keys'],
    kernel_size=15,
    peak_thresh=1.96,
    group_threshold=2,
    time_variance='peak_time',
    subsample=False,
    scramble_stim_ids=True
)

single_trial_data=single_trial_data_standard_non_stimd_V1_all_cells

filtered_array_V1,a,result_V1_scrambled = calculate_single_trial_features.process_and_filter_response_matrix(
    dff_trials=single_trial_data['trig_dff_trials'],
    roi_ids=single_trial_data['roi_ids'],
    stim_ids=single_trial_data['stim_ids'],
    roi_keys=single_trial_data['roi_keys'],
    kernel_size=15,
    peak_thresh=1.96,
    group_threshold=2,
    time_variance='peak_time',
    subsample=False,
    scramble_stim_ids=True
)
# %%

# ---- Common Setup ----
features_to_track = [
    'peak_time_std', 'com_std', 'auc_avg', 'peak_array_mean_trial', 'peak_amp_avg',
    'peak_time_array_mean_trial', 'com_array_mean_trial', 'response_proportion',
    'com_calc_abs_diff_mean_trial', 'peak_time_calc_abs_diff_mean_trial'
]

# ---- Define filters ----
filter_steps = [
    ('response_proportion >= 0.6', lambda df: df[df['response_proportion'] >= 0.6]),
    # ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) < 3]),
    ('peak_array_mean_trial > 0', lambda df: df[df['peak_array_mean_trial'] > 2]),
    ('peak_time_std <= 0.5', lambda df: df[df['peak_time_std'] <= 0.5]),
    # ('peak_time_avg <= 0.5', lambda df: df[df['peak_time_avg'] >= 1.0]),
    # ('peak_time_calc_abs_diff_mean_trial <= 0.4', lambda df: df[df['peak_time_calc_abs_diff_mean_trial'] <= 0.2]),
    ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 2]),
    ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 0]),
]
# %%
result_M2['is_shuffled'] = False
result_M2_filt, summaries_M2 = calculate_single_trial_features.filter_and_summarize(result_M2, result_M2, filter_steps, features_to_track, label="real")
# ---- Process real ----
result_M2_scrambled['is_shuffled'] = False
result_M2_scrambled_filt, summaries_scrambled_M2 = calculate_single_trial_features.filter_and_summarize(result_M2_scrambled, result_M2_scrambled, filter_steps, features_to_track, label="real")

result_V1['is_shuffled'] = False
result_V1_filt, summaries_V1 = calculate_single_trial_features.filter_and_summarize(result_V1, result_V1, filter_steps, features_to_track, label="real")
# ---- Process real ----
result_V1_scrambled['is_shuffled'] = False
result_V1_scrambled_filt, summaries_scrambled_V1 = calculate_single_trial_features.filter_and_summarize(result_V1_scrambled, result_V1_scrambled, filter_steps, features_to_track, label="real")
# %%


calculate_single_trial_features.prepare_and_plot_heatmap(result_V1_scrambled_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-2,
    vmax=2,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None
    )

calculate_single_trial_features.prepare_and_plot_heatmap(result_M2_scrambled_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-2,
    vmax=2,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None)