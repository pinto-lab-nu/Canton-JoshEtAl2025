# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 14:00:08 2025

@author: jec822
"""

# %%

# Define the columns to match on
group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

# Merge the two DataFrames on the group columns
merged_df = result_V1_standard_non_stimd_filt.merge(
    summary_stats_V1_standard_stimd[group_cols + ['responsive_cells_per_roi']],
    on=group_cols,
    how='left'
)

# Filter for responsive_cell_per_roi == 0
filtered_result = merged_df[merged_df['responsive_cells_per_roi'] == 1].copy()

# (Optional) Drop the 'responsive_cell_per_roi' column if you want only the original columns from result_M2_standard_stimd_filt
filtered_result = filtered_result[result_M2_standard_stimd_filt.columns]
# %%

calculate_single_trial_features.prepare_and_plot_heatmap(result_M2_short_stim_non_stimd_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=-2,
    vmax=2,
    start_index=0,
    sampling_interval=0.032958316,
    exclude_window=None
    )
