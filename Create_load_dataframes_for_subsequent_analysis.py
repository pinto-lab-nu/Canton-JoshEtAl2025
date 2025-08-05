# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 14:24:00 2025

@author: jec822
"""

import os
import joblib
from analyzeSpont2P import params as tau_params
import analyzeSpont2P
from analyzeEvoked2P import params as opto_params
import analyzeEvoked2P
import Canton_Josh_et_al_2025_analysis_plotting_functions as analysis_plotting_functions
import General_functions_Calculate_responsive_cells_create_pseudo_trial_arrays as calculate_single_trial_features
import Cross_Correlation_functions as calculate_cross_corr

# %%

def fetch_or_load_single_trial_data(
    area,
    params,
    expt_type='standard',
    resp_type='dff',
    eg_ids=None,
    signif_only=False,
    which_neurons='non_stimd',
    relax_timing_criteria=False,
    save_data=True,
    date_str='',
    data_dir='.'
):
    """
    Fetches or loads single trial data for a given brain area and condition.
    Also returns the corresponding variable name.
    """
    import analyzeEvoked2P

    # Construct base name and filename
    var_name = f"single_trial_data__{area}_{expt_type}_{which_neurons}_sig_{signif_only}"
    fname = f"{var_name}_{date_str}.joblib"
    fpath = os.path.join(data_dir, fname)

    # Load or compute
    if os.path.exists(fpath):
        print(f"[LOAD] Loading data from {fpath}")
        data = joblib.load(fpath)
    else:
        print(f"[COMPUTE] Computing data for {var_name}")
        data = analyzeEvoked2P.get_single_trial_data(
            area=area,
            params=params,
            expt_type=expt_type,
            resp_type=resp_type,
            eg_ids=eg_ids,
            signif_only=signif_only,
            which_neurons=which_neurons,
            relax_timing_criteria=relax_timing_criteria
        )
        if save_data:
            joblib.dump(data, fpath)
            print(f"[SAVE] Saved data to {fpath}")

    return data, var_name


# %%

def process_and_summarize_single_trial_data(
    area_str,
    opto_params,
    expt_type,
    which_neurons,
    signif_only,
    filter_steps,
    features_to_track,
    save_data=True,
    date_str='',
    return_data=False,
    max_trials_per_group=20
):
    # Fetch or load trial data
    data, varname = fetch_or_load_single_trial_data(
        area=area_str,
        params=opto_params,
        expt_type=expt_type,
        which_neurons=which_neurons,
        signif_only=signif_only,
        save_data=save_data,
        date_str=date_str
    )
    globals()[varname] = data

    # Extract area and remove suffix if needed
    area = varname.split("__")[1]
    area_clean = area.removesuffix("_sig_False")

    # Get trial data
    single_trial_data = globals()[varname]

    # Process response matrix
    _, _, result = calculate_single_trial_features.process_and_filter_response_matrix(
        dff_trials=single_trial_data['trig_dff_trials'],
        roi_ids=single_trial_data['roi_ids'],
        stim_ids=single_trial_data['stim_ids'],
        roi_keys=single_trial_data['roi_keys'],
        time_axes=single_trial_data['time_axis_sec'],
        kernel_size=7,
        peak_thresh=3,
        group_threshold=2,
        time_variance='peak_time',
        subsample=False,
        scramble_stim_ids=False,
        min_width_val=2,
        max_trials_per_group=max_trials_per_group
    )
    # Add metadata
    # result= add_roi_metadata_individual_fetches(result)
    
    
    # Save result
    globals()[f"result_{area_clean}"] = result

    # Filter and summarize
    result_filt, summaries = calculate_single_trial_features.filter_and_summarize(
        result, result, filter_steps, features_to_track, label="real"
    )

    # Add metadata
    result_filt = add_roi_metadata_individual_fetches(result_filt)

    # Save filtered results and summaries
    globals()[f"result_{area_clean}_filt"] = result_filt
    globals()[f"summaries_{area_clean}"] = summaries

    # Summary stats
    summary_stats = summarize_responsive_proportion(result, result_filt)
    globals()[f"summary_stats_{area_clean}"] = summary_stats

    # Return everything
    if return_data:
        return area_clean, single_trial_data, result, result_filt, summaries, summary_stats
    else:
        return area_clean, result, result_filt, summaries, summary_stats



# %%
def summarize_responsive_proportion(df_all, df_filt, area=None):
    import pandas as pd
    import inspect

    # Try to infer area name from the df_filt variable name if not provided
    if area is None:
        callers_globals = inspect.currentframe().f_back.f_globals
        for varname, val in callers_globals.items():
            if isinstance(val, pd.DataFrame) and val is df_filt and varname.startswith("result_") and "_filt" in varname:
                area = varname.replace("result_", "").replace("_filt", "")
                break
        if area is None:
            raise ValueError("Area name could not be inferred. Please pass `area='V1'`, etc. explicitly.")

    # Grouping columns
    group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

    # Count total cells
    counts_all = df_all.groupby(group_cols).size().reset_index(name='cells_per_roi')

    # Count responsive cells
    counts_filt = df_filt.groupby(group_cols).size().reset_index(name='responsive_cells_per_roi')

    # Compute average peak_time_avg for responsive cells
    peak_time_avg = (
        df_filt.groupby(group_cols)['peak_time_avg']
        .mean()
        .reset_index(name='peak_time_avg')
    )

    # Merge everything
    merged_counts = counts_all.merge(counts_filt, on=group_cols, how='left')
    merged_counts = merged_counts.merge(peak_time_avg, on=group_cols, how='left')

    # Fill missing values
    merged_counts['responsive_cells_per_roi'] = merged_counts['responsive_cells_per_roi'].fillna(0).astype(int)
    merged_counts['peak_time_avg'] = merged_counts['peak_time_avg'].fillna(float('nan'))

    # Calculate proportion
    merged_counts['responsive_proportion'] = (
        merged_counts['responsive_cells_per_roi'] / merged_counts['cells_per_roi']
    )

    # Save to global variable
    varname = f"summary_stats_{area}"
    callers_globals[varname] = merged_counts

    return merged_counts


# %% Very slow but only needs to be done once, can be sped up significantly but makes for messier code
# import numpy as np

# def add_roi_metadata_individual_fetches(df, roi_key_col='roi_keys'):
#     from schemas import spont_timescales
#     from schemas import twop_opto_analysis
#     df = df.copy()

#     fields_to_fetch = {
#         'tau': spont_timescales.TwopTau,
#         'snr': VM['twophoton'].Snr2P,
#         'events_per_min': VM['twophoton'].Snr2P,
#         'min_dist_from_stim_um': twop_opto_analysis.TrigDffTrialAvg,
#         'is_stimd': twop_opto_analysis.TrigDffTrialAvg,
#         'max_or_min_dff': twop_opto_analysis.TrigDffTrialAvg,
#         'time_of_peak_sec_poststim': twop_opto_analysis.TrigDffTrialAvg,
#         'time_of_trough_sec_poststim': twop_opto_analysis.TrigDffTrialAvg,
#         'is_good_tau':spont_timescales.TwopTauInclusion
#     }

#     for col_name, table in fields_to_fetch.items():
#         values = []
#         for roi in df[roi_key_col]:
#             try:
#                 val = (table & roi).fetch1(col_name)
#             except:
#                 val = np.nan
#             values.append(val)
#         df[col_name] = values

#     return df

# %%
def add_roi_metadata_individual_fetches(df, roi_key_col='roi_keys'):
    from schemas import spont_timescales
    from schemas import twop_opto_analysis
    df = df.copy()

    # Map desired column name -> (table, actual field name)
    fields_to_fetch = {
        'tau': (spont_timescales.TwopTau, 'tau'),
        'snr': (VM['twophoton'].Snr2P, 'snr'),
        'events_per_min': (VM['twophoton'].Snr2P, 'events_per_min'),
        'min_dist_from_stim_um': (twop_opto_analysis.TrigDffTrialAvg, 'min_dist_from_stim_um'),
        'is_stimd': (twop_opto_analysis.TrigDffTrialAvg, 'is_stimd'),
        'max_or_min_dff': (twop_opto_analysis.TrigDffTrialAvg, 'max_or_min_dff'),
        'time_of_peak_sec_poststim': (twop_opto_analysis.TrigDffTrialAvg, 'time_of_peak_sec_poststim'),
        'time_of_trough_sec_poststim': (twop_opto_analysis.TrigDffTrialAvg, 'time_of_trough_sec_poststim'),
        'is_good_tau': (spont_timescales.TwopTauInclusion, 'is_good_tau_roi')  # Adjust field name as needed
    }

    for col_name, (table, field_name) in fields_to_fetch.items():
        values = []
        for roi in df[roi_key_col]:
            try:
                val = (table & roi).fetch1(field_name)
            except:
                val = np.nan
            values.append(val)
        df[col_name] = values

    return df


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
    ('peak_time_std <= 0.75', lambda df: df[df['peak_time_std'] <= 0.75]),
    # ('peak_time_avg <= 0.5', lambda df: df[df['peak_time_avg'] >= 1.0]),
    # ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 1.96]),
    ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 0])
]
# %%

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='M2',
    opto_params=opto_params,
    expt_type='standard',
    which_neurons='non_stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True
)

# %%

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='M2',
    opto_params=opto_params,
    expt_type='standard',
    which_neurons='stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True
)
# %%

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='V1',
    opto_params=opto_params,
    expt_type='standard',
    which_neurons='non_stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True
)

# %%

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='V1',
    opto_params=opto_params,
    expt_type='standard',
    which_neurons='stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True
)

# %% High trial count M2

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='M2',
    opto_params=opto_params,
    expt_type='high_trial_count',
    which_neurons='non_stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True,
    max_trials_per_group=20
)

# %% 1 spiral M2 stimd

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='M2',
    opto_params=opto_params,
    expt_type='standard',
    which_neurons='stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True
)

# %%1 spiral M2 non stimd


area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='M2',
    opto_params=opto_params,
    expt_type='short_stim',
    which_neurons='non_stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True
)

# %%

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='M2',
    opto_params=opto_params,
    expt_type='short_stim',
    which_neurons='stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True
)

# %% Multi_cell M2

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='M2',
    opto_params=opto_params,
    expt_type='multi_cell',
    which_neurons='non_stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True
)

# Multi_cell V1

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='V1',
    opto_params=opto_params,
    expt_type='multi_cell',
    which_neurons='non_stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True
)
# %% This is the same as the function call above but it is expanded

# SAVE_DATA = True
# DATE_STR = ''

# # M2 non-stimulated
# data, varname = fetch_or_load_single_trial_data(
#     area='M2',
#     params=opto_params,
#     expt_type='standard',
#     which_neurons='non_stimd',
#     signif_only=False,
#     save_data=SAVE_DATA,
#     date_str=DATE_STR
# )
# globals()[varname] = data

# # Process it and name result automatically as result_{area}
# area = varname.split("__")[1]  # Extract area from varname
# single_trial_data = globals()[varname]

# _, _, result = calculate_single_trial_features.process_and_filter_response_matrix(
#     dff_trials=single_trial_data['trig_dff_trials'],
#     roi_ids=single_trial_data['roi_ids'],
#     stim_ids=single_trial_data['stim_ids'],
#     roi_keys=single_trial_data['roi_keys'],
#     time_axes=single_trial_data['time_axis_sec'],
#     kernel_size=7,
#     peak_thresh=2.96,
#     group_threshold=2,
#     time_variance='peak_time',
#     subsample=False,
#     scramble_stim_ids=False,
#     min_width_val=2
# )

# # Save result with automatic name like result_V1
# globals()[f"result_{area}"] = result

# area = area.removesuffix("_sig_False")

# result_filt, summaries = calculate_single_trial_features.filter_and_summarize(
#     result, result, filter_steps, features_to_track, label="real"
# )

# # Save using result_{area}_filt and summaries_{area}
# globals()[f"result_{area}_filt"] = calculate_single_trial_features.add_roi_metadata_individual_fetches(result_filt)
# globals()[f"summaries_{area}"] = summaries


# summary_stats = summarize_responsive_proportion(
#     result,
#     result_filt
# )

# globals()[f"summary_stats_{area}"] = summary_stats



