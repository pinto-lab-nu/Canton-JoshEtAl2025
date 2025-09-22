# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:42:45 2025

@author: jec822
"""


trial_data=result_M2_standard_non_stimd

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
    # ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 2]),
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

# %%
result_M2_filt, summaries_M2 = calculate_single_trial_features.filter_and_summarize(result_M2_standard_non_stimd, result_M2_standard_non_stimd, filter_steps, features_to_track, label="pseudo")

# %% Figure 4G


xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
    # peak_or_com_series=result_M2_standard_non_stimd['peak_time_by_trial'],
    peak_or_com_series=result_M2_filt['peak_time_by_trial'],
    metric='peak',
    n_iter=5
    # random_seed=42
)

analyzeEvoked2P.plot_trial_xval(area='M2', 
                                                     params=opto_params, 
                                                     expt_type='standard', 
                                                     resp_type='dff', 
                                                     signif_only=True, 
                                                     which_neurons='non_stimd', 
                                                     # axis_handle=row3_axs[0],
                                                     axis_handle=None,
                                                     # fig_handle=fig,
                                                     trial_data=None,
                                                     xval_results=xval_results)
       
# %%
xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
    peak_or_com_series=result_M2_pseudo_filt['peak_time_by_trial'],
    metric='peak',
    n_iter=1,
    # random_seed=42
)


analyzeEvoked2P.plot_trial_xval(area='M2', 
                                                     params=opto_params, 
                                                     expt_type='standard', 
                                                     resp_type='dff', 
                                                     signif_only=True, 
                                                     which_neurons='non_stimd', 
                                                     # axis_handle=row3_axs[0],
                                                     axis_handle=None,
                                                     # fig_handle=fig,
                                                     trial_data=None,
                                                     xval_results=xval_results)
       
# %%

xval_summary = []
iterations_xval = 1000

for i in range(iterations_xval):
    xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
        peak_or_com_series=result_M2_filt['peak_time_by_trial'],
        metric='peak',
        n_iter=5,  # single iteration per call
        random_seed=None  # or set = i for reproducibility
    )

    # Extract r and r² values
    xval_summary.append({
        'iteration': i,
        'correlation_r': xval_results['correlation_r'],
        'correlation_r2': xval_results['correlation_r2']
    })

# Convert to DataFrame
xval_df_high_trial = pd.DataFrame(xval_summary)


xval_summary = []
iterations_xval = 1000

for i in range(iterations_xval):
    xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
        peak_or_com_series=result_M2_pseudo_filt['peak_time_by_trial'],
        metric='peak',
        n_iter=5,  # single iteration per call
        random_seed=None  # or set = i for reproducibility
    )

    # Extract r and r² values
    xval_summary.append({
        'iteration': i,
        'correlation_r': xval_results['correlation_r'],
        'correlation_r2': xval_results['correlation_r2']
    })

# Convert to DataFrame
xval_df = pd.DataFrame(xval_summary)

# %%

analysis_plotting_functions.plot_ecdf_comparison(xval_df['correlation_r'],xval_df_high_trial['correlation_r'],
                     label1='"MOs 5 trial pseudo_data', label2="MOs 5 trial data",
                     line_color1='k',line_color2='k',
                     xlabel="r values",
                     ylabel="proportion of iterations",
                     title='',
                     stat_test='auto',
                     figsize=[4,4],
                      xticks_start=0, xticks_end=1, xticks_step=.2,
                       xlim=(0,.5)
                     # yticks_start=0, yticks_end=1, yticks_step=0.5,
                     # show_normality_pvals=True,
                     )

# %%
# n_iter_list=[5]
# # n_iter_list=[1, 5, 10, 50]

# xval_nested_dfs_high_trial = calculate_cross_corr.run_xval_across_iterations(
#     series_data=result_M2_high_trial_count_non_stimd_filt['peak_time_by_trial'],
#     n_iter_list=n_iter_list,
#     iter_total=1000,
#     metric='peak',
#     base_seed=42,
#     verbose=True
# )
# # 

# xval_nested_dfs = calculate_cross_corr.run_xval_across_iterations(
#     series_data=result_M2_pseudo_filt['peak_time_by_trial'],
#     n_iter_list=n_iter_list,
#     iter_total=1000,
#     metric='peak',
#     base_seed=42,
#     verbose=True
# )

# # %%

# r_matrix_df_real = pd.DataFrame({n_iter: df['correlation_r'].values for n_iter, df in xval_nested_dfs_high_trial.items()})

# r_matrix_df_pseudo = pd.DataFrame({n_iter: df['correlation_r'].values for n_iter, df in xval_nested_dfs.items()})


# distances = list(r_matrix_df_real.columns)
# distances_scaled = [val / max(distances) for val in distances]


# # %% Figure 4H

# analysis_plotting_functions.plot_ecdf_comparison(r_matrix_df_real[5],r_matrix_df_pseudo[5],
#                      label1='"MOs high trial count pseudo_data', label2="MOs high trial count data",
#                      line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
#                      xlabel="r values",
#                      ylabel="proportion of iterations",
#                      title='',
#                      stat_test='auto',
#                      figsize=[3,4],
#                      # xticks_start=0, xticks_end=25, xticks_step=5,
#                      # yticks_start=0, yticks_end=1, yticks_step=0.5,
#                      show_normality_pvals=True,
#                      )



