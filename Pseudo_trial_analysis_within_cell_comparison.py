# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:47:34 2025

@author: jec822
"""

# %% This generate pseudo trial form responsive cells only

trial_data=result_V1_standard_non_stimd_filt

pseudo_dfs_iteration_V1=calculate_single_trial_features.generate_pseudo_trials_select_cells(
    roi_ids = np.array(trial_data['roi_id_extended_dataset']),
    stim_ids = np.array(trial_data['stim_id']),
    roi_keys_all = np.array(trial_data['roi_keys']),
    trials = trial_data['trials_arrays'],
    n_iters=1000,
    baseline_window=(1000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80,
    repeat_count=10
)

joblib.dump(pseudo_dfs_iteration_V1, 'pseudo_trial_within_cell_control_V1_1000_iter_20250724.joblib')

# pseudo_dfs_iteration_V1=joblib.load('pseudo_trial_within_cell_control_V1_1000_iter_20250724.joblib')


 # This is for the cell by cell pseudo trials single iteration

# pseudo_trials_dffs_iter_V1=pseudo_dfs_iteration_V1['iter_0']

# result_V1_pseudo= calculate_single_trial_features.process_and_filter_response_matrix_from_df(
#     entire_df=pseudo_trials_dffs_iter_V1,
#     kernel_size=7,
#     peak_thresh=2.96,
#     group_threshold=2,
#     time_variance='peak_time', # or False to use COM
#     subsample=False,
#     min_width_val=5
# )
# this repeats over the 1000 repeats of the within pseudo population

# Dictionary to hold result_dfs for each iteration
pseudo_results_multi_iter_V1 = {}

# Loop over all iterations in your pseudo dataframe dictionary
for iter_label, pseudo_trials_dffs_iter in pseudo_dfs_iteration_V1.items():

    # Run your processing function
    result_V1_pseudo= calculate_single_trial_features.process_and_filter_response_matrix_from_df(
        entire_df=pseudo_trials_dffs_iter,
        kernel_size=7,
        peak_thresh=2.94,
        group_threshold=2,
        time_variance='peak_time', # or False to use COM
        subsample=False,
        min_width_val=2
    )
    
    # Store the result DataFrame in the dictionary
    pseudo_results_multi_iter_V1[iter_label] =  result_V1_pseudo
    
# 
joblib.dump(pseudo_results_multi_iter_V1, 'results_pseudo_trial_within_cell_control_V1_1000_iter_20250724.joblib')
# %%

pseudo_results_filt_multi_iter_V1 = {}

# Loop over all iterations in your pseudo dataframe dictionary
for iter_label, pseudo_trials_dffs_iter in pseudo_results_multi_iter_V1.items():

    # Run your processing function
    result_V1_pseudo_filt, summaries_V1_pseudo = calculate_single_trial_features.filter_and_summarize(pseudo_trials_dffs_iter, pseudo_trials_dffs_iter, filter_steps, features_to_track, label="pseudo")

    
    # Store the result DataFrame in the dictionary
    pseudo_results_filt_multi_iter_V1[iter_label] =  result_V1_pseudo_filt
    
# %%

pseudo_results_filt_multi_iter_V1=joblib.load('results_pseudo_trial_within_cell_control_V1_1000_iter_20250724.joblib')

# %%
result_V1_wihtin_cell_pseudo=pseudo_results_filt_multi_iter_V1['iter_0']
# %%
from collections import Counter

# Dictionary to track how many iterations each ROI appears in
roi_iteration_counts = Counter()

# Loop through each iteration
for df in pseudo_results_filt_multi_iter_V1.values():
    unique_rois = df['roi_id_extended_dataset'].unique()
    roi_iteration_counts.update(unique_rois)

# Convert to DataFrame
roi_iteration_df = pd.DataFrame({
    'roi_id': list(roi_iteration_counts.keys()),
    'iteration_count': list(roi_iteration_counts.values())
})


# Plot histogram of the counts
plt.figure(figsize=(6, 4))
plt.hist(roi_iteration_df ['iteration_count'], bins=20, edgecolor='black')
plt.xlabel('Count of ROI occurrences')
plt.ylabel('Frequency')
plt.title('Histogram of ROI Counts')
plt.tight_layout()
plt.show()
# %%  THis counts across all iterations (multipe er iteration possible)
combined_df = pd.concat(pseudo_results_filt_multi_iter_V1.values(), axis=0, ignore_index=True)


threshold = 50  # Filter on count, not roi_id value

# Get the Series
roi_ids = combined_df['roi_id_extended_dataset']

# Count occurrences of each roi_id
counts = roi_ids.value_counts().sort_index()

# Filter to keep only those with count > threshold
filtered_counts = counts[counts > threshold]

# Convert to labeled DataFrame
df_result = pd.DataFrame({
    'roi_id': filtered_counts.index,
    'count': filtered_counts.values
})

# %%
import matplotlib.pyplot as plt

# Plot histogram of the counts
plt.figure(figsize=(6, 4))
plt.hist(df_result['count'], bins=20, edgecolor='black')
plt.xlabel('Count of ROI occurrences')
plt.ylabel('Frequency')
plt.title('Histogram of ROI Counts')
plt.tight_layout()
plt.show()
# %%
filtered_result_V1_filt = result_V1_filt[~result_V1_filt['roi_id_extended_dataset'].isin(df_result['roi_id'])]


# %% This generate pseudo trial form responsive cells only M2 Version

trial_data=result_M2_standard_non_stimd_filt

pseudo_dfs_iteration_M2=calculate_single_trial_features.generate_pseudo_trials_select_cells(
    roi_ids = np.array(trial_data['roi_id_extended_dataset']),
    stim_ids = np.array(trial_data['stim_id']),
    roi_keys_all = np.array(trial_data['roi_keys']),
    trials = trial_data['trials_arrays'],
    n_iters=1000,
    baseline_window=(1000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80,
    repeat_count=10
)

joblib.dump(pseudo_dfs_iteration_M2, 'pseudo_trial_within_cell_control_M2_1000_iter_20250724.joblib')


# 
# pseudo_dfs_iteration_M2=joblib.load('pseudo_trial_within_cell_control_M2_1000_iter_20250724.joblib')
# 
# pseudo_trials_dffs_iter_M2=pseudo_dfs_iteration_M2['iter_100']


# This is for the cell by cell pseudo trials
# result_M2_pseudo= calculate_single_trial_features.process_and_filter_response_matrix_from_df(
#     entire_df=pseudo_trials_dffs_iter_M2,
#     kernel_size=7,
#     peak_thresh=2.96,
#     group_threshold=2,
#     time_variance='peak_time', # or False to use COM
#     subsample=False,
#     min_width_val=5
# )

# this repeats over the 100 repeats of the within pseudo population

# Dictionary to hold result_dfs for each iteration
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
    
# %%
joblib.dump(pseudo_results_multi_iter_M2, 'results_pseudo_trial_within_cell_control_M2_1000_iter_20250724.joblib')

# %%
pseudo_results_filt_multi_iter_M2 = {}

# Loop over all iterations in your pseudo dataframe dictionary
for iter_label, pseudo_trials_dffs_iter in pseudo_results_multi_iter_M2.items():

    # Run your processing function
    result_M2_pseudo_filt, summaries_M2_pseudo = calculate_single_trial_features.filter_and_summarize(pseudo_trials_dffs_iter, pseudo_trials_dffs_iter, filter_steps, features_to_track, label="pseudo")

    
    # Store the result DataFrame in the dictionary
    pseudo_results_filt_multi_iter_M2[iter_label] =  result_M2_pseudo_filt


# %%
from collections import Counter

# Dictionary to track how many iterations each ROI appears in
roi_iteration_counts = Counter()

# Loop through each iteration
for df in pseudo_results_filt_multi_iter_M2.values():
    unique_rois = df['roi_id_extended_dataset'].unique()
    roi_iteration_counts.update(unique_rois)

# Convert to DataFrame
roi_iteration_df = pd.DataFrame({
    'roi_id': list(roi_iteration_counts.keys()),
    'iteration_count': list(roi_iteration_counts.values())
})


# Plot histogram of the counts
plt.figure(figsize=(6, 4))
plt.hist(roi_iteration_df ['iteration_count'], bins=20, edgecolor='black')
plt.xlabel('Count of ROI occurrences')
plt.ylabel('Frequency')
plt.title('Histogram of ROI Counts')
plt.tight_layout()
plt.show()
# %%  THis counts across all iterations (multipe er iteration possible)
combined_df = pd.concat(pseudo_results_filt_multi_iter_M2.values(), axis=0, ignore_index=True)


threshold = 150  # Filter on count, not roi_id value

# Get the Series
roi_ids = combined_df['roi_id_extended_dataset']

# Count occurrences of each roi_id
counts = roi_ids.value_counts().sort_index()

# Filter to keep only those with count > threshold
filtered_counts = counts[counts > threshold]

# Convert to labeled DataFrame
df_result = pd.DataFrame({
    'roi_id': filtered_counts.index,
    'count': filtered_counts.values
})
# %%
import matplotlib.pyplot as plt

# Plot histogram of the counts
plt.figure(figsize=(6, 4))
plt.hist(df_result['count'], bins=20, edgecolor='black')
plt.xlabel('Count of ROI occurrences')
plt.ylabel('Frequency')
plt.title('Histogram of ROI Counts')
plt.tight_layout()
plt.show()
# %%
filtered_result_M2_filt = result_M2_filt[~result_M2_filt['roi_id_extended_dataset'].isin(df_result['roi_id'])]

# %%
plt.figure()
plt.scatter(non_overlapping_df_sorted['peak_time_avg'].to_numpy(),non_overlapping_df_sorted['roi_occurrence_all'].to_numpy())

plt.figure()
plt.scatter(non_overlapping_df_sorted['peak_time_avg'].to_numpy(),non_overlapping_df_sorted['peak_time_std'].to_numpy())


