# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:38:13 2025

@author: jec822
"""

# %%

pseudo_dfs_iteration_M2_sub_sampled=calculate_single_trial_features.generate_pseudo_trials_subsampled(
    single_trial_data_standard_non_stimd_M2_all_cells,
    n_iters=1,
    baseline_window=(5000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80
)

# %%
import pandas as pd

pseudo_trial_data=pseudo_trials_list_M2
single_trial_data=single_trial_data_standard_non_stimd_M2_all_cells

# Define number of iterations
n_iters = 1000

# Output dictionary to store result DataFrames
pseudo_results_iterations_M2 = {}

for i in range(n_iters):
    print(f"Processing iteration {i+1}/{n_iters}...")

    _, _, result_df = calculate_single_trial_features.process_and_filter_response_matrix(
        dff_trials=pseudo_trial_data,
        roi_ids=single_trial_data['roi_ids'],
        stim_ids=single_trial_data['stim_ids'],
        roi_keys=single_trial_data['roi_keys'],
        kernel_size=15,
        peak_thresh=1.96,
        group_threshold=2,
        time_variance='peak_time',
        subsample=True,
        n_subsample=250,         # set sample size
        random_seed=i            # ensure different sample each time
    )

    pseudo_results_iterations_M2[f"iter_{i}"] = result_df
    
    
pseudo_trial_data=pseudo_trials_list_V1
single_trial_data=single_trial_data_standard_non_stimd_V1_all_cells

pseudo_results_iterations_V1 = {}

for i in range(n_iters):
    print(f"Processing iteration {i+1}/{n_iters}...")

    _, _, result_df = calculate_single_trial_features.process_and_filter_response_matrix(
        dff_trials=pseudo_trial_data,
        roi_ids=single_trial_data['roi_ids'],
        stim_ids=single_trial_data['stim_ids'],
        roi_keys=single_trial_data['roi_keys'],
        kernel_size=15,
        peak_thresh=1.96,
        group_threshold=2,
        time_variance='peak_time',
        subsample=True,
        n_subsample=250,         # set sample size
        random_seed=i            # ensure different sample each time
    )

    pseudo_results_iterations_V1[f"iter_{i}"] = result_df

# import joblib

# joblib.dump(pseudo_results_iterations_M2, 'pseudo_iterations_M2_sub_sampled_250_cells_1000_iterations.joblib')
# joblib.dump(pseudo_results_iterations_V1, 'pseudo_iterations_V1_sub_sampled_250_cells_1000_iterations.joblib')

# %%
import joblib

pseudo_results_iterations_M2=joblib.load('pseudo_iterations_M2_sub_sampled_250_cells_1000_iterations.joblib')
pseudo_results_iterations_V1=joblib.load('pseudo_iterations_V1_sub_sampled_250_cells_1000_iterations.joblib')

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
    # ('peak_array_mean_trial > 0', lambda df: df[df['peak_array_mean_trial'] > 2]),
    ('peak_time_std <= 0.5', lambda df: df[df['peak_time_std'] <= 1.0]),
    # ('peak_time_calc_abs_diff_mean_trial <= 0.4', lambda df: df[df['peak_time_calc_abs_diff_mean_trial'] <= 0.2]),
    ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 2]),
    ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 0]),
]
# %%

# Output dictionary to store each iteration's summary DataFrame
pseudo_summaries_iterations_M2_subsampled = {}
pseudo_results_iterations=pseudo_results_iterations_M2

n_iters = 1000

# Loop over each iteration
for i in range(n_iters):
    iter_key = f"iter_{i}"
    
    # Extract iteration-specific pseudo trial DFFs
    result_pseudo_trials_dffs_iter= pseudo_results_iterations[iter_key]

    # Create DataFrame like result_M2_pseudo for this iteration
    result_M2_pseudo_subsampled = result_pseudo_trials_dffs_iter.copy()  # or however your real structure is defined
    result_M2_pseudo_subsampled['is_shuffled'] = True

    # Run filtering & summarizing pipeline
    result_M2_pseudo_subsampled_filt,summaries_M2_pseudo_subsampled = calculate_single_trial_features.filter_and_summarize(
        result_M2_pseudo_subsampled,
        result_M2_pseudo_subsampled,
        filter_steps,
        features_to_track,
        label="pseudo"
    )

    # Save summary to dictionary
    pseudo_summaries_iterations_M2_subsampled[iter_key] = summaries_M2_pseudo_subsampled
    
# Output dictionary to store each iteration's summary DataFrame
pseudo_summaries_iterations_V1_subsampled = {}
pseudo_results_iterations=pseudo_results_iterations_V1

n_iters = 1000

# Loop over each iteration
for i in range(n_iters):
    iter_key = f"iter_{i}"
    
    # Extract iteration-specific pseudo trial DFFs
    result_pseudo_trials_dffs_iter= pseudo_results_iterations[iter_key]

    # Create DataFrame like result_V1_pseudo for this iteration
    result_V1_pseudo_subsampled = result_pseudo_trials_dffs_iter.copy()  # or however your real structure is defined
    result_V1_pseudo_subsampled['is_shuffled'] = True

    # Run filtering & summarizing pipeline
    result_V1_pseudo_subsampled_filt,summaries_V1_pseudo_subsampled = calculate_single_trial_features.filter_and_summarize(
        result_V1_pseudo_subsampled,
        result_V1_pseudo_subsampled,
        filter_steps,
        features_to_track,
        label="pseudo"
    )

    # Save summary to dictionary
    pseudo_summaries_iterations_V1_subsampled[iter_key] = summaries_V1_pseudo_subsampled
# %%
import pandas as pd

# Get the filtering step names from the first summary
filter_steps_names = pseudo_summaries_iterations['iter_0']['step'].tolist()

# Initialize a dictionary to hold reshaped results per step
reshaped_dict = {step: [] for step in filter_steps}

# Loop through iterations and collect each row i into the corresponding step
for i in range(len(pseudo_summaries_iterations_M2_subsampled)):
    iter_df = pseudo_summaries_iterations_M2_subsampled[f"iter_{i}"]
    for step in filter_steps_names:
        step_row = iter_df[iter_df['step'] == step].copy()
        step_row['iteration'] = i
        reshaped_dict[step].append(step_row)

# Concatenate each list into a single DataFrame per step
reshaped_step_dfs_M2 = {
    step: pd.concat(step_rows, ignore_index=True)
    for step, step_rows in reshaped_dict.items()
}

# Initialize a dictionary to hold reshaped results per step
reshaped_dict = {step: [] for step in filter_steps}

# Loop through iterations and collect each row i into the corresponding step
for i in range(len(pseudo_summaries_iterations_V1_subsampled)):
    iter_df = pseudo_summaries_iterations_V1_subsampled[f"iter_{i}"]
    for step in filter_steps_names:
        step_row = iter_df[iter_df['step'] == step].copy()
        step_row['iteration'] = i
        reshaped_dict[step].append(step_row)

# Concatenate each list into a single DataFrame per step
reshaped_step_dfs_V1 = {
    step: pd.concat(step_rows, ignore_index=True)
    for step, step_rows in reshaped_dict.items()
}
# %%

calculate_single_trial_features.prepare_and_plot_heatmap(result_M2_pseudo_subsampled_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-2,
    vmax=2,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None
    )

calculate_single_trial_features.prepare_and_plot_heatmap(result_V1_pseudo_subsampled_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-2,
    vmax=2,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None)