# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:32:04 2025

@author: jec822
"""

trial_data = get_single_trial_data(area='M2', expt_type='high_trial_count',
                                   params=params,
                                    resp_type='dff', 
                                    signif_only=True, 
                                    which_neurons='stimd', 
                                    # axis_handle=row3_axs[0],
                                    # relax_timing_criteria=params['xval_relax_timing_criteria']
                                    relax_timing_criteria=False
                                    )

# %% peak time analysis , feed a lot of the code need to put bakc into the fucntion but biggest issue is it finds signifcance even duirng 
# baseline

analyze_baseline=False
sampling_interval=0.032958316
rng = np.random.default_rng(seed=params['random_seed'])
sem_overall  = list()
median_half1 = list()
median_half2 = list()
unique_rois  = list(np.unique(trial_data['roi_ids']))

baseline_idx  = (0, 80)
stimulus_idx  = (100, 380)

# This was created to keep track of trials to make sure iterations is working well
avg_dffs=pd.DataFrame()

pseudo_trials_dffs=pd.DataFrame()

for roi in unique_rois:
    ridx = trial_data['roi_ids'] == roi
    
    # Load raw values
    these_trials = np.array(trial_data['trial_ids'])[ridx]
    these_stims  = np.array(trial_data['stim_ids'])[ridx]
    coms         = np.array(trial_data['com_sec'])[ridx]
    peaks        = np.array(trial_data['peak_or_trough_time_sec'])[ridx]
    t_axes       = np.array(trial_data['time_axis_sec'], dtype=object)[ridx]

    roi_keys     =list(np.array(trial_data['roi_keys'])[ridx])
    
    
    # Step 1: Get all trials
    trials = trial_data['trig_dff_trials']
    axes = trial_data['time_axis_sec']
    # Step 2: Find the minimum length among them
    min_len = min([len(trial) for trial in trials])
    # Step 3: Truncate all trials to that length and stack into a 2D array
    a = np.array([trial[:min_len] for trial in trials])
    axes_temp=np.array([ax[:min_len] for ax in axes])
    
    trial_dffs_all=list(a)
    
    t_axes=list(axes_temp)
    
    unique_stims = np.unique(these_stims)
    
    # take random halves of trials and compare timing stats for each 
    for stim in unique_stims:
        sidx = trial_data['stim_ids'] == stim
        tidx= np.where(ridx & sidx)[0]
        
        # tidx        = these_trials[these_stims==stim]-1
        tidx_shuff  = deepcopy(tidx)
        ntrials     = np.size(tidx)
        timing_set1 = np.zeros(params['xval_num_iter'])
        timing_set2 = np.zeros(params['xval_num_iter'])
        # timing_set1 = np.zeros(5)
        # timing_set2 = np.zeros(5)
        taxis       = t_axes[tidx[0]]
        frame_int   = np.diff(taxis)[0]
    
        dffs = [trial_dffs_all[i] for i in tidx] 
        roi = roi_keys[0]
        
        
        mean_trace = pd.DataFrame(dffs).mean(axis=0)

        # Append the mean as a new row
        avg_dffs = pd.concat([avg_dffs, pd.DataFrame([mean_trace])], ignore_index=True)
        
        # baseline_dff = (VM['twophoton'].Dff2P & roi).fetch('dff')[0][0]
        baseline_dff = (VM['twophoton'].Rawf2P & roi).fetch('rawf')[0][0]
        baseline_dff = baseline_dff[0:10000]
        # baseline_dff = (baseline_dff - np.nanmean(baseline_dff)) / np.nanstd(baseline_dff)
        
        trial_length = dffs[0].shape[0]  # assuming all real trials have the same length
        
        pseudo_trials = create_pseudo_trials_from_baseline_non_overlaping(
            baseline_dff=baseline_dff,
            number_of_trials=len(dffs),
            trial_length=trial_length,
            baseline_start_idx=0,
            max_offset=10000,
            zscore_to_baseline=True,
            # baseline_window_for_z=80
        )
        
        pseudo_trials_dffs = pd.concat([pseudo_trials_dffs, pd.DataFrame(pseudo_trials)], ignore_index=True)
        
        if analyze_baseline:
            peak_array_m2, auc_array_m2, fhwm_m2, peak_times, com_values, com_times = process_trig_dff_trials(
                # trial_dffs,
                pseudo_trials,
                kernel_size=15,
                peak_window=[stimulus_idx[0]-stimulus_idx[0], stimulus_idx[1]-stimulus_idx[0]],
                use_prominence=False,
                prominence_val=1.96,
                trace_start=stimulus_idx[0]
            )
        else:    
            peak_array_m2, auc_array_m2, fhwm_m2, peak_times, com_values, com_times = process_trig_dff_trials(
                # trial_dffs,
                dffs,
                kernel_size=15,
                peak_window=[stimulus_idx[0]-stimulus_idx[0], stimulus_idx[1]-stimulus_idx[0]],
                use_prominence=False,
                prominence_val=1.96,
                trace_start=stimulus_idx[0]
            )
        
        peak_threshold = 1.96
        peak_times = np.array(peak_times, dtype=float) * sampling_interval
        com_times = np.array(com_times, dtype=float) * sampling_interval
        
        # NaN out values where peak amplitude is too low
        peak_times[np.array(peak_array_m2) < peak_threshold] = np.nan
        com_times[np.array(peak_array_m2) < peak_threshold] = np.nan
        
        # Keep only trials where peak or com is valid, depending on metric
       
        valid_mask = ~np.isnan(peak_times)
        
        
        # Require >50% of trials to be valid
        valid_indices = np.where(valid_mask)[0]
        valid_frac = len(valid_indices) / ntrials
        
        if valid_frac <= 0.5:
            print(f"Skipping stim {stim}: only {valid_frac*100:.1f}% valid trials")
            continue  # skip this stim
        
        # Now shuffle only among valid trials
        for iShuff in range(params['xval_num_iter']):
            shuff_idx = deepcopy(valid_indices)
            rng.shuffle(shuff_idx)
            n_half = len(shuff_idx) // 2
            half1 = shuff_idx[:n_half]
            half2 = shuff_idx[n_half:]
        
            if params['xval_timing_metric'] == 'peak':
                timing_set1[iShuff] = np.nanmedian(peak_times[half1])
                timing_set2[iShuff] = np.nanmedian(peak_times[half2])
                if iShuff == 0:
                    sem_overall.append(np.nanstd(peak_times[valid_indices]) / np.sqrt(len(valid_indices) - 1))
        
            elif params['xval_timing_metric'] == 'com':
                timing_set1[iShuff] = np.nanmedian(com_times[half1])
                timing_set2[iShuff] = np.nanmedian(com_times[half2])
                if iShuff == 0:
                    sem_overall.append(np.nanstd(com_times[valid_indices]) / np.sqrt(len(valid_indices) - 1))

            # # or just take a median of pre-computed trial-by-trial features    
            # # else:
            # if params['xval_timing_metric'] == 'peak':
            #     timing_set1[iShuff] = np.nanmedian(peaks[half1])
            #     timing_set2[iShuff] = np.nanmedian(peaks[half2])
            #     if iShuff == 0:
            #         sem_overall.append(np.std(peaks)/np.sqrt(len(tidx)-1))
                
            # elif params['xval_timing_metric'] == 'com':
            #     timing_set1[iShuff] = np.nanmedian(coms[half1])
            #     timing_set2[iShuff] = np.nanmedian(coms[half2])
            #     if iShuff == 0:
            #         sem_overall.append(np.std(coms)/np.sqrt(len(tidx)-1))
                
            # else:
            #     print('unknown parameter value for timing metric, returning nothing')
            #     # return None
        
        # collect median metric for each half
        median_half1.append(np.nanmedian(timing_set1))
        median_half2.append(np.nanmedian(timing_set2))
        
xval_results = {
            'timing_metric'     : params['xval_timing_metric'], 
            'trial_sem'         : np.array(sem_overall).flatten(), 
            'median_trialset1'  : np.array(median_half1).flatten(),
            'median_trialset2'  : np.array(median_half2).flatten(), 
            'response_type'     : 'dff', 
            'experiment_type'   : 'standard', 
            'which_neurons'     : 'non_stimd',
            'analysis_params'   : deepcopy(params)
            }
# %%

analyzeEvoked2P.plot_trial_xval(area='M2', 
                                                     params=opto_params, 
                                                     expt_type='standard', 
                                                     resp_type='dff', 
                                                     signif_only=True, 
                                                     which_neurons='non_stimd', 
                                                     # axis_handle=row3_axs[0],
                                                     axis_handle=None,
                                                     # fig_handle=fig,
                                                     trial_data=trial_data,
                                                     xval_results=xval_results)
       
# %% Kind of nice but unneccesary since there are buolt in repeats in ishuff
# analyze_baseline=True
# sampling_interval=0.032958316
# rng = np.random.default_rng(seed=params['random_seed'])
# sem_overall  = list()
# median_half1 = list()
# median_half2 = list()
# unique_rois  = list(np.unique(trial_data['roi_ids']))

# baseline_idx  = (0, 80)
# stimulus_idx  = (100, 380)
    

# n_repeats = 5  # or whatever number of repetitions you want

# for roi in unique_rois:
#     ridx = trial_data['roi_ids'] == roi
#     these_trials = np.array(trial_data['trial_ids'])[ridx]
#     these_stims  = np.array(trial_data['stim_ids'])[ridx]
#     t_axes       = np.array(trial_data['time_axis_sec'], dtype=object)[ridx]
#     roi_keys     = list(np.array(trial_data['roi_keys'])[ridx])

#     trials = trial_data['trig_dff_trials']
#     axes = trial_data['time_axis_sec']
#     min_len = min([len(trial) for trial in trials])
#     a = np.array([trial[:min_len] for trial in trials])
#     trial_dffs_all = list(a)
#     t_axes = list(np.array([ax[:min_len] for ax in axes]))

#     unique_stims = np.unique(these_stims)

#     for stim in unique_stims:
#         sidx = trial_data['stim_ids'] == stim
#         tidx = np.where(ridx & sidx)[0]
#         dffs = [trial_dffs_all[i] for i in tidx]
#         roi = roi_keys[0]

#         baseline_dff = (VM['twophoton'].Rawf2P & roi).fetch('rawf')[0][0][:10000]
#         trial_length = dffs[0].shape[0]

#         all_medians_half1 = []
#         all_medians_half2 = []

#         for rep in range(n_repeats):
#             pseudo_trials = create_pseudo_trials_from_baseline(
#                 baseline_dff=baseline_dff,
#                 number_of_trials=len(dffs),
#                 trial_length=trial_length,
#                 baseline_start_idx=0,
#                 max_offset=3000,
#                 zscore_to_baseline=True,
#                 baseline_window_for_z=80
#             )

#             dff_input = pseudo_trials if analyze_baseline else dffs

#             peak_array_m2, _, _, peak_times, _, com_times = process_trig_dff_trials(
#                 dff_input,
#                 kernel_size=15,
#                 peak_window=[0, stimulus_idx[1] - stimulus_idx[0]],
#                 use_prominence=False,
#                 prominence_val=1.96,
#                 trace_start=stimulus_idx[0]
#             )

#             peak_times = np.array(peak_times) * sampling_interval
#             com_times = np.array(com_times) * sampling_interval
#             peak_times[np.array(peak_array_m2) < 1.96] = np.nan
#             com_times[np.array(peak_array_m2) < 1.96] = np.nan

#             valid_indices = np.where(~np.isnan(peak_times))[0]
            
#             if len(valid_indices) < 2:
#                 continue

#             timing_set1 = np.zeros(params['xval_num_iter'])
#             timing_set2 = np.zeros(params['xval_num_iter'])

#             for i in range(params['xval_num_iter']):
#                 shuff_idx = deepcopy(valid_indices)
#                 rng.shuffle(shuff_idx)
#                 n_half = len(shuff_idx) // 2
#                 half1, half2 = shuff_idx[:n_half], shuff_idx[n_half:]

#                 if params['xval_timing_metric'] == 'peak':
#                     timing_set1[i] = np.nanmedian(peak_times[half1])
#                     timing_set2[i] = np.nanmedian(peak_times[half2])
#                 elif params['xval_timing_metric'] == 'com':
#                     timing_set1[i] = np.nanmedian(com_times[half1])
#                     timing_set2[i] = np.nanmedian(com_times[half2])
#                 else:
#                     raise ValueError('Unknown timing metric')

#             all_medians_half1.append(np.nanmedian(timing_set1))
#             all_medians_half2.append(np.nanmedian(timing_set2))

#         median_half1.append(np.nanmean(all_medians_half1))
#         median_half2.append(np.nanmean(all_medians_half2))

# %% Cross correlation with trial pairs

import pandas as pd

baseline_idx  = (0, 80)
stimulus_idx  = (100, 380)
unique_rois  = list(np.unique(trial_data['roi_ids']))

results = []  # list to collect per-row dictionaries

for roi in unique_rois:
    ridx = trial_data['roi_ids'] == roi
    
    # Load raw values
    these_trials = np.array(trial_data['trial_ids'])[ridx]
    these_stims  = np.array(trial_data['stim_ids'])[ridx]
    coms         = np.array(trial_data['com_sec'])[ridx]
    peaks        = np.array(trial_data['peak_or_trough_time_sec'])[ridx]
    t_axes       = np.array(trial_data['time_axis_sec'], dtype=object)[ridx]

    roi_keys     =list(np.array(trial_data['roi_keys'])[ridx])
    
    # Step 1: Get all trials
    trials = trial_data['trig_dff_trials']
    axes = trial_data['time_axis_sec']
    # Step 2: Find the minimum length among them
    min_len = min([len(trial) for trial in trials])
    # Step 3: Truncate all trials to that length and stack into a 2D array
    a = np.array([trial[:min_len] for trial in trials])
    axes_temp=np.array([ax[:min_len] for ax in axes])
    
    trial_dffs_all=list(a)
    
    t_axes=list(axes_temp)
    
    unique_stims = np.unique(these_stims)
    
    # take random halves of trials and compare timing stats for each 
    for stim in unique_stims:
        sidx = trial_data['stim_ids'] == stim
        tidx= np.where(ridx & sidx)[0]
        # dffs = [trial_dffs[i] for i in stim_mask]
        dffs = [trial_dffs_all[i] for i in tidx] 
        roi = roi_keys[0]
        
        # baseline_dff = (VM['twophoton'].Dff2P & roi).fetch('dff')[0][0]
        baseline_dff = (VM['twophoton'].Rawf2P & roi).fetch('rawf')[0][0]
        baseline_dff = baseline_dff[0:10000]
        # baseline_dff = (baseline_dff - np.nanmean(baseline_dff)) / np.nanstd(baseline_dff)
        
        trial_length = dffs[0].shape[0]  # assuming all real trials have the same length
        
        # pseudo_trials = create_pseudo_trials_from_baseline(
        #     baseline_dff=baseline_dff,
        #     number_of_trials=len(dffs),
        #     trial_length=trial_length,
        #     baseline_start_idx=0,
        #     max_offset=3000,
        #     zscore_to_baseline=True,
        #     baseline_window_for_z=80
        # )
        
        peak_array_m2, auc_array_m2, fhwm_m2, peak_times, com_values, com_times = process_trig_dff_trials(
            dffs,
            kernel_size=15,
            peak_window=[0, 250],
            use_prominence=False,
            prominence_val=1.96,
            trace_start=100
        )

        peak_threshold = 1.96
        peak_times = np.array(peak_times, dtype=float) * sampling_interval
        com_times  = np.array(com_times, dtype=float) * sampling_interval

        peak_times[np.array(peak_array_m2) < peak_threshold] = np.nan
        com_times[np.array(peak_array_m2) < peak_threshold] = np.nan

        valid_mask = ~np.isnan(peak_times)
        valid_indices = np.where(valid_mask)[0]
        ntrials = len(dffs)
        valid_frac = len(valid_indices) / ntrials
        
        if valid_frac >= 0.5:
            _,cross_corr_df,_, _ = cross_correlation_with_mean_random(
                dffs, 
                start_index=stimulus_idx[0], 
                end_index=stimulus_idx[1], 
                number_of_trials=1, 
                kernel_size=9
            )
            
            all_mean_corrs = []
            
            for _ in range(10):  # 10 rounds
                pseudo_trials = create_pseudo_trials_from_baseline(baseline_dff=baseline_dff,
                    number_of_trials=len(dffs),
                    trial_length=trial_length,
                    baseline_start_idx=0,
                    max_offset=3000,
                    zscore_to_baseline=True,
                    within_trial_baseline_frames=80
                )
                
                
                
                _, cross_corr_df_baseline, _, _ = cross_correlation_with_mean_random(
                    pseudo_trials, 
                    start_index=stimulus_idx[0], 
                    end_index=stimulus_idx[1], 
                    number_of_trials=1, 
                    kernel_size=9
                )
                
                cross_corr_df_baseline = cross_corr_df_baseline[cross_corr_df_baseline.max(axis=1) > 0]
                all_mean_corrs.append(cross_corr_df_baseline.mean())
            
            mean_corr_baseline = pd.concat(all_mean_corrs, axis=1).mean(axis=1)
            
            # _,cross_corr_df_baseline, _, _ = cross_correlation_with_mean_random(
            #     pseudo_trials, 
            #     start_index=stimulus_idx[0], 
            #     end_index=stimulus_idx[1], 
            #     number_of_trials=1, 
            #     kernel_size=9
            # )
            
            # Remove rows in which all values are zero or max is 0
            cross_corr_df = cross_corr_df[cross_corr_df.max(axis=1) > 0]
            # cross_corr_df_baseline = cross_corr_df_baseline[cross_corr_df_baseline.max(axis=1) > 0]
            
            # Now compute the means
            mean_corr = cross_corr_df.mean()
            # mean_corr_baseline = cross_corr_df_baseline.mean()

            # Save row to results
            results.append({
                'roi': roi,
                'stim': stim,
                'mean_cross_corr': mean_corr,
                'mean_cross_corr_baseline': mean_corr_baseline
            })

# Convert to DataFrame and save
cross_corr_summary_df_baseline = pd.DataFrame(results)

# %%
a = cross_corr_summary_df_baseline["mean_cross_corr"]

# Stack arrays into a 2D numpy array
array_2d = np.vstack(a.values)

# plt.plot(np.mean(array_2d,axis=0))
# Convert to DataFrame (optional: keep index from original)
df_2d = pd.DataFrame(array_2d, index=cross_corr_summary_df_baseline.index)

plt.plot(df_2d.mean())
# %%

a = cross_corr_summary_df_baseline["mean_cross_corr_baseline"]

# Stack arrays into a 2D numpy array
array_2d = np.vstack(a.values)

# Convert to DataFrame (optional: keep index from original)
df_2d_baseline = pd.DataFrame(array_2d, index=cross_corr_summary_df_baseline.index)

plt.plot(df_2d_baseline.mean())

# %% this now works with output from new analysis in pseudo trials pipeline

cross_corr_summary_df_M2_filt = calculate_cross_corr.compute_cross_correlation_summary(result_M2_filt,number_of_trials=2,stimulus_idx=(100, 380))

cross_corr_summary_df_M2_filt_late = calculate_cross_corr.compute_cross_correlation_summary(filtered_result_M2_filt,number_of_trials=2,stimulus_idx=(100, 380))

cross_corr_summary_df_M2_pseudo_filt = calculate_cross_corr.compute_cross_correlation_summary(result_M2_pseudo,number_of_trials=2,stimulus_idx=(100, 380))

# cross_corr_summary_df_M2_scrambled_filt = compute_cross_correlation_summary(result_M2_scrambled_filt,number_of_trials=2,stimulus_idx=(100, 380))

# cross_corr_summary_df_M2_pseudo_subsampled = compute_cross_correlation_summary(result_M2_pseudo_subsampled,number_of_trials=2,stimulus_idx=(100, 380))

# %%

cross_corr_summary_df_M2_high_filt = calculate_cross_corr.compute_cross_correlation_summary(result_M2_high_filt,number_of_trials=10,stimulus_idx=(100, 380))

cross_corr_summary_df_M2_high_pseudo_filt = calculate_cross_corr.compute_cross_correlation_summary(result_M2_pseudo_high_filt,number_of_trials=10,stimulus_idx=(100, 380))


# %%

analysis_plotting_functions.plot_cross_correlation_means(
    summary_dfs=[cross_corr_summary_df_M2_filt ,
                 cross_corr_summary_df_M2_pseudo_filt,
                 cross_corr_summary_df_M2_filt_late
                   # cross_corr_summary_df_M2_scrambled_filt,
                   # cross_corr_summary_df_M2_pseudo_subsampled
    ],
    labels=["MOs responder all",
            "MOs Pseudo qll",
            'MOs responder late',
            # 'MOs Pseudo subsampled'
            ],
    # colors=["black"],
    # alphas=[0.7]
)
# %%
# Compute cross-correlation summaries for V1 datasets
cross_corr_summary_df_V1_filt = calculate_cross_corr.compute_cross_correlation_summary(result_V1_standard_non_stimd_filt)

# cross_corr_summary_df_V1_filt_late = compute_cross_correlation_summary(result_V1_filt_late)

cross_corr_summary_df_V1_pseudo_filt = calculate_cross_corr.compute_cross_correlation_summary(result_V1_pseudo_filt)

# cross_corr_summary_df_V1_scrambled_filt = compute_cross_correlation_summary(result_V1_scrambled_filt)

cross_corr_summary_df_V1_pseudo_subsampled = calculate_cross_corr.compute_cross_correlation_summary(result_V1_wihtin_cell_pseudo)
# %%

# Plot V1 cross-correlation curves
analysis_plotting_functions.plot_cross_correlation_means(
    summary_dfs=[
        cross_corr_summary_df_V1_filt,
        cross_corr_summary_df_V1_pseudo_filt,
        # cross_corr_summary_df_M2_filt,
        # cross_corr_summary_df_V1_scrambled_filt,
        cross_corr_summary_df_V1_pseudo_subsampled
    ],
    labels=["V1 responder all",
            "V1 Pseudo all",
            'V1 pseudo within'],
    # labels=["V1"],
    # colors=["black"],
    # alphas=[0.7]
)

# %%

def stack_lag_dataframes(df, lags_column='lags',column_label='Lag'):
    lag_arrays = []

    for idx, lag_df in df[lags_column].items():
        # Flatten to 1D array
        if isinstance(lag_df, pd.DataFrame) and 'Lag' in lag_df.columns:
            lag_array = lag_df[column_label].values
        else:
            continue  # skip malformed or missing lags

        lag_arrays.append(pd.Series(lag_array, name=idx))

    # Combine into a DataFrame: rows = ROI-stim entries, cols = trial lags
    stacked_lags_df = pd.DataFrame(lag_arrays)

    return stacked_lags_df

def stack_nested_dataframes(df, nested_column='lags', value_column='Lags'):
    """
    Stack a column of nested DataFrames into a 2D DataFrame.

    Parameters:
        df : pd.DataFrame
            DataFrame with a column of DataFrames (e.g., lags or peaks).
        nested_column : str
            Column name in df containing the nested DataFrames.
        value_column : str
            Column name inside the nested DataFrames to extract (e.g., 'Lag' or 'peak').

    Returns:
        pd.DataFrame
            A DataFrame where each row corresponds to a parent row in df,
            and columns correspond to the extracted values from the nested DataFrames.
    """
    arrays = []

    for idx, nested_df in df[nested_column].items():
        if isinstance(nested_df, pd.DataFrame) and value_column in nested_df.columns:
            values = nested_df[value_column].values
            arrays.append(pd.Series(values, name=idx))
        else:
            continue  # skip malformed or missing data

    stacked_df = pd.DataFrame(arrays)

    return stacked_df


# %%

lags_matrix_real = stack_lag_dataframes(cross_corr_summary_df_M2_filt_late)
lags_matrix_pseudo = stack_lag_dataframes(cross_corr_summary_df_M2_pseudo_filt)
# lags_matrix_scrambled = stack_lag_dataframes(cross_corr_summary_df_M2_scrambled_filt)

peaks_matrix_real = stack_nested_dataframes(cross_corr_summary_df_M2_filt_late,nested_column='peaks',value_column='Peaks')
peaks_matrix_pseudo = stack_nested_dataframes(cross_corr_summary_df_M2_pseudo_filt,nested_column='peaks',value_column='Peaks')

stacked_peaks_df_real = np.array(pd.concat(cross_corr_summary_df_M2_filt_late['peaks'].tolist(), ignore_index=True))
stacked_peaks_df_pseudo = np.array(pd.concat(cross_corr_summary_df_M2_pseudo_filt['peaks'].tolist(), ignore_index=True))


# %%
plt.figure()
sns.histplot(lags_matrix_real.values.flatten(), label='Real', color='black', stat='density')
plt.figure()
sns.histplot(lags_matrix_pseudo.values.flatten(), label='Pseudo', color='blue', stat='density')
# %%
# Real
mean_lags_real = lags_matrix_real.mean(axis=1)
std_lags_real = lags_matrix_real.std(axis=1)

# Pseudo
mean_lags_pseudo = lags_matrix_pseudo.mean(axis=1)
std_lags_pseudo = lags_matrix_pseudo.std(axis=1)

# Scrambled
# mean_lags_scrambled = lags_matrix_scrambled.mean(axis=1)
# std_lags_scrambled = lags_matrix_scrambled.std(axis=1)


# Real
mean_peaks_real = peaks_matrix_real.mean(axis=1, skipna=True)
std_peaks_real = peaks_matrix_real.std(axis=1, skipna=True)

# Pseudo
mean_peaks_pseudo = peaks_matrix_pseudo.mean(axis=1, skipna=True)
std_peaks_pseudo = peaks_matrix_pseudo.std(axis=1, skipna=True)

# %%
analysis_plotting_functions.plot_ecdf_comparison(lags_matrix_real.values.flatten(),lags_matrix_pseudo.values.flatten(),
                     label1="real data late", label2="pseudo data",title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak z-score",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=300, xticks_step=50,
                     xlim=[-300,300],
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True)

# %%
analysis_plotting_functions.plot_ecdf_comparison(mean_lags_real,mean_lags_pseudo,
                     label1="real data late", label2="pseudo data",title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak z-score",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=300, xticks_step=50,
                     # xlim=[-300,300],
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True)

# %%

analysis_plotting_functions.plot_ecdf_comparison(mean_peaks_real,mean_peaks_pseudo,
                     label1="real data late", label2="pseudo data",title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak z-score",
                     ylabel="Directly stimulated cells",
                     # xticks_start=0, xticks_end=300, xticks_step=50,
                     # xlim=[-300,300],
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True)

# %%
analysis_plotting_functions.plot_ecdf_comparison(stacked_peaks_df_real,stacked_peaks_df_pseudo,
                     label1="real data late", label2="pseudo data",title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak z-score",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=300, xticks_step=50,
                     # xlim=[-300,300],
                     stat_test='auto',
                     # figsize=[3,4],
                     show_normality_pvals=True)
# %%
analysis_plotting_functions.plot_ecdf_comparison(std_lags_real,std_lags_pseudo,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak z-score",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=300, xticks_step=50,
                     # xlim=[-300,300],
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True)

# %%

xval_results = crossval_peak_timing_shuffle(
    peak_or_com_series=result_M2_pseudo_filt['peak_times_by_trial'],
    metric='peak',
    n_iter=100,
    random_seed=42
)

# %%

xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
    peak_or_com_series=result_M2_standard_non_stimd_filt['peak_time_by_trial'],
    metric='peak',
    n_iter=1,
    random_seed=42
)
# %%

xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
    peak_or_com_series=result_V1_standard_non_stimd_filt['peak_time_by_trial'],
    metric='peak',
    n_iter=1,
    random_seed=42
)

# %%
xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
    peak_or_com_series=result_V1_wihtin_cell_pseudo['peak_time_by_trial'],
    metric='peak',
    n_iter=1,
    random_seed=42
)

# %%
xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
    peak_or_com_series=result_V1_pseudo['peak_time_by_trial'],
    metric='peak',
    n_iter=1,
    random_seed=42
)


# %%
xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
    peak_or_com_series=result_V1_pseudo_filt['peak_time_by_trial'],
    metric='peak',
    n_iter=1,
    random_seed=42
)


# %%
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
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def compute_icc2_manual(peak_times_by_trial):
    """
    Computes ICC(2,1) using a random-effects ANOVA model from statsmodels,
    which works with unbalanced data and removes NaNs per group.

    Parameters:
    - peak_times_by_trial: pd.Series, where each row contains a list/array of peak times (including NaNs)

    Returns:
    - icc: float, ICC(2,1)
    """
    # Create long-form DataFrame
    records = []
    for i, trial_list in enumerate(peak_times_by_trial):
        for j, val in enumerate(trial_list):
            if not np.isnan(val):
                records.append({
                    'cell_stim': f"unit_{i}",
                    'trial': f"trial_{j}",
                    'peak_time': val
                })

    df = pd.DataFrame(records)

    # Use mixed-effects model to estimate variance components
    model = smf.mixedlm("peak_time ~ 1", df, groups=df["cell_stim"])
    result = model.fit()

    # Extract variance components
    var_between = result.cov_re.iloc[0, 0]  # between cell_stim
    var_within = result.scale               # residual error

    icc = var_between / (var_between + var_within)

    return icc


# %%
icc_value = compute_icc2_manual(result_M2['peak_times_by_trial'])
print(f"ICC(2,1): {icc_value:.3f}")