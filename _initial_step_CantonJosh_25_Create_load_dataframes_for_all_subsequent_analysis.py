# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 14:24:00 2025

@author: jec822
"""


# =============================
# =========  Set up ===========
# =============================

# ======  Dependencies  ========
# PintoLab_dj 
# PintoLab_imagingAnalysis
# PintoLab_utils
# these can all be installed as packages

# ======  Import stuff  ========
from utils import connect_to_dj
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
from scipy.signal import medfilt
import pandas as pd
from schemas import spont_timescales
from schemas import twop_opto_analysis

VM = connect_to_dj.get_virtual_modules()

# ======= Local modules ========
# code_dir = "/Users/lpr6177/Documents/code/Canton-JoshEtAl2025/"
code_dir = "/Users/Jec822/Documents/GitHub/Canton-JoshEtAl2025"
sys.path.insert(0,code_dir)
from analyzeSpont2P import params as tau_params
import analyzeSpont2P
from analyzeEvoked2P import params as opto_params
import analyzeEvoked2P
import Canton_Josh_et_al_2025_analysis_plotting_functions as analysis_plotting_functions
import Canton_Josh_et_al_2025_PCA_functions as PCA_functions
import General_functions_Calculate_responsive_cells_create_pseudo_trial_arrays as calculate_single_trial_features
import Cross_Correlation_functions as calculate_cross_corr
import matplotlib as mpl


# Set global matplotlib rcParams
new_rc_params = {
    'text.usetex': False,          # Disable LaTeX rendering
    'svg.fonttype': 'none',        # Keep text as text in SVG
    'pdf.fonttype': 42,            # Use Type 42 (TrueType) fonts in PDF
    'ps.fonttype': 42,             # Use Type 42 (TrueType) fonts in PS
    'font.family': 'Arial'         # Use Arial font
}
mpl.rcParams.update(new_rc_params)

def letter_annotation(ax, xoffset, yoffset, letter):
    ax.text(xoffset, yoffset, letter, transform=ax.transAxes,
            size=12, weight='bold', family='Arial')  # Arial font explicitly set here too
 



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
    # var_name = f"single_trial_data__{area}_{expt_type}_{which_neurons}_sig_{signif_only}_resp_type_{resp_type}"
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
import os
import joblib

def save_results_bundle(area_clean, date_str, result, result_filt, summaries, folder="results_cache"):
    """Save result, result_filt, and summaries together using joblib."""
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/results_bundle_{area_clean}_{date_str}.joblib"
    joblib.dump(
        {
            "result": result,
            "result_filt": result_filt,
            "summaries": summaries
        },
        filename
    )
    print(f"[Saved bundle] {filename}")

def load_results_bundle(area_clean, date_str, folder="results_cache"):
    """Load result, result_filt, and summaries bundle if exists."""
    filename = f"{folder}/results_bundle_{area_clean}_{date_str}.joblib"
    if os.path.exists(filename):
        bundle = joblib.load(filename)
        print(f"[Loaded bundle] {filename}")
        return bundle["result"], bundle["result_filt"], bundle["summaries"]
    return None, None, None

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
    max_trials_per_group=20,
    peak_window=[0.3,10],
    add_metadata_pre_filter=False,
    load_existing=True,
    apply_new_filter=False,   # <--- NEW FLAG
    save_folder="results_cache",
    resp_type='dff',
    inhibited_cells=False,
    peak_thresh=2.94
):
    # Fetch or load trial data
    data, varname = fetch_or_load_single_trial_data(
        area=area_str,
        params=opto_params,
        expt_type=expt_type,
        which_neurons=which_neurons,
        signif_only=signif_only,
        save_data=save_data,
        date_str=date_str,
        resp_type=resp_type
    )
    globals()[varname] = data

    # Extract area and clean name
    area = varname.split("__")[1]
    area_clean = area.removesuffix("_sig_False")
    
    # Add suffix depending on inhibited_cells
    if inhibited_cells:
        area_clean += "_inh"
    else:
        area_clean += "_exc"

    # Try loading precomputed results
    result, result_filt, summaries = None, None, None
    if load_existing:
        result, result_filt, summaries = load_results_bundle(area_clean, date_str, folder=save_folder)

    # Case 1: No cached result available → compute everything
    if result is None or (not load_existing):
        # Process response matrix
        _, _, result = calculate_single_trial_features.process_and_filter_response_matrix(
            dff_trials=data['trig_dff_trials'],
            roi_ids=data['roi_ids'],
            stim_ids=data['stim_ids'],
            roi_keys=data['roi_keys'],
            time_axes=data['time_axis_sec'],
            kernel_size=7,
            peak_thresh=peak_thresh,
            group_threshold=2,
            time_variance='peak_time', 
            subsample=False,
            scramble_stim_ids=False,
            min_width_val=2,
            max_trials_per_group=max_trials_per_group,
            peak_window_sec=peak_window,
            inhibited_cells=inhibited_cells
        )
        
        
        if add_metadata_pre_filter:
            
            result = add_roi_metadata_individual_fetches(result)
        

            group_cols = ['subject_fullname', 'session_date', 'scan_number', 'roi_id']
            col_map = {
                'tau':'taus',
                'autocorr_vals': 'autocorr_vals',
                'lags_sec': 'lags',
                'time_vector': 'time_vector',
                'dual_fit_params': 'dual_fit_params',
                'r2_fit_single': 'r2_s',
                'bic_single': 'bic_s',
                'r2_fit_double': 'r2_d',
                'bic_double': 'bic_d',
                'acorr_1_index': 'acorr_1_index'
            }
            result = process_roi_keys_and_add_fits(result, group_cols, col_map)
            result = result.rename(columns={"taus": "tau"})

            result = result.drop_duplicates(subset=["stim_id", "roi_id_extended_dataset"])


    # Case 2: Apply new filtering even if loaded from cache
    if result is not None and (result_filt is None or summaries is None or apply_new_filter):
        result_filt, summaries = calculate_single_trial_features.filter_and_summarize(
            result, result, filter_steps, features_to_track, label="real"
        )

        if not add_metadata_pre_filter:
            result_filt = add_roi_metadata_individual_fetches(result_filt)

        if save_data:
            save_results_bundle(area_clean, date_str, result, result_filt, summaries, folder=save_folder)


    # Save to globals
    globals()[f"result_{area_clean}"] = result
    globals()[f"result_{area_clean}_filt"] = result_filt
    globals()[f"summaries_{area_clean}"] = summaries

    # Summary stats
    summary_stats = summarize_responsive_proportion(result, result_filt)
    globals()[f"summary_stats_{area_clean}"] = summary_stats

    if return_data:
        return area_clean, data, result, result_filt, summaries, summary_stats
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

# %%
def add_roi_metadata_individual_fetches(df, roi_key_col='roi_keys'):
    from schemas import spont_timescales
    from schemas import twop_opto_analysis
    df = df.copy()

    # Map desired column name -> (table, actual field name)
    fields_to_fetch = {
        # 'tau': (spont_timescales.TwopTau, 'tau'),
        'snr': (VM['twophoton'].Snr2P, 'snr'),
        'events_per_min': (VM['twophoton'].Snr2P, 'events_per_min'),
        'min_dist_from_stim_um': (twop_opto_analysis.TrigDffTrialAvg, 'min_dist_from_stim_um'),
        'is_stimd': (twop_opto_analysis.TrigDffTrialAvg, 'is_stimd'),
        'max_or_min_dff': (twop_opto_analysis.TrigDffTrialAvg, 'max_or_min_dff'),
        'time_of_peak_sec_poststim': (twop_opto_analysis.TrigDffTrialAvg, 'time_of_peak_sec_poststim'),
        'time_of_trough_sec_poststim': (twop_opto_analysis.TrigDffTrialAvg, 'time_of_trough_sec_poststim'),
        # 'is_good_tau': (spont_timescales.TwopTauInclusion, 'is_good_tau_roi')  # Adjust field name as needed
    }


    
    for col_name, (table, field_name) in fields_to_fetch.items():
        values = []
        for roi in df[roi_key_col]:
            try:
                val = (table & roi[0]).fetch1(field_name)
            except:
                val = np.nan
            values.append(val)
        df[col_name] = values

    return df

# %%  This could definitely be sped up by really only need to do it once ...

def fetch_roi_metadata_from_keys(roi_keys):
    from schemas import spont_timescales
    from schemas import twop_opto_analysis
    import numpy as np
    import pandas as pd

    # Create initial DataFrame
    df = pd.DataFrame({'roi_keys': roi_keys})

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
        'is_good_tau': (spont_timescales.TwopTauInclusion, 'is_good_tau_roi')
    }

    col_map = {}  # Dictionary to store column: (table, field_name)

    for col_name, (table, field_name) in fields_to_fetch.items():
        values = []
        for roi in roi_keys:
            try:
                val = (table & roi).fetch1(field_name)
            except:
                val = np.nan
            values.append(val)
        df[col_name] = values
        col_map[col_name] = (table.__name__, field_name)

    return df, col_map

# %%
def add_columns(df_source, df_target, group_cols, col_map):
    """
    Merge multiple tau columns from df_source into df_target based on shared group columns.

    Parameters:
    - df_source: DataFrame containing the original tau values.
    - df_target: DataFrame to which the tau values will be added.
    - group_cols: List of columns to merge on.
    - col_map: Dictionary mapping original column names in df_source to new column names in df_target.

    Returns:
    - df_target with new columns added.
    """
    tau_df = df_source[group_cols + list(col_map.keys())].copy()
    tau_df = tau_df.rename(columns=col_map)
    return df_target.merge(tau_df, on=group_cols, how='left')


def process_roi_keys_and_add_fits(result_df, group_cols, col_map,dff_type='residuals_dff'):
    import numpy as np
    from schemas import spont_timescales
    import analyzeSpont2P

    # Extract ROI keys
    keys = result_df['roi_keys'].reset_index(drop=True)

    corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
    param_key         = {'corr_param_set_id': corr_param_set_id}
 
    # Fetch matching keys from spont_timescales
    k_fetched = (spont_timescales.TwopAutocorr & keys & param_key).fetch('KEY')

    # Extract fit data
    df_fits = analyzeSpont2P.extract_fit_data_for_keys_df(k_fetched)

    # Add acorr[1] index value
    df_fits['acorr_1_index'] = [
        vec[1] if isinstance(vec, (list, np.ndarray)) and len(vec) > 1 else np.nan
        for vec in df_fits['autocorr_vals']
    ]

    # Merge back into original DataFrame
    return add_columns(df_fits, result_df, group_cols, col_map)
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
    ('peak_time_std <= 0.75', lambda df: df[df['peak_time_std'] <=0.75]),
    ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 0])
]
# %% standard M2 non-stimd

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
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=True,
    peak_window=[0.25,8.25]
)

# Standard V1_nonstimd

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
    return_data=True,
    add_metadata_pre_filter=False,
    apply_new_filter=True,
    peak_window=[0.25,8.25]

)

# %%

 #  standard M2 stimd

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
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=False,
    peak_window=[0.25,8.25]
)

# Standard V1_stimd

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
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=False,
    peak_window=[0.25,8.25]
)

# 
#  1 spiral v1 non stimd

area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='V1',
    opto_params=opto_params,
    expt_type='short_stim',
    which_neurons='non_stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True,
    peak_window=[0.00,8.25],
    add_metadata_pre_filter=True,
    apply_new_filter=False,
    
)

#  1 spiral M2 non stimd


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
    return_data=True,
    peak_window=[0.00,8.25],
    add_metadata_pre_filter=True,
    apply_new_filter=False
)
# 
# 1 spiral data stimd

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
    return_data=True,
    peak_window=[0.00,8.25],
    add_metadata_pre_filter=True,
    apply_new_filter=False
)


area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='V1',
    opto_params=opto_params,
    expt_type='short_stim',
    which_neurons='stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True,
    peak_window=[0.0,8.25],
    add_metadata_pre_filter=True,
    apply_new_filter=False
)

# %%
filter_steps = [
    ('response_proportion >= 0.6', lambda df: df[df['response_proportion'] >= 0.6]),

]
#  High trial count M2

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
    max_trials_per_group=20,
    apply_new_filter=False
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


# %% standard M2 non-stimd dconvolved

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
    return_data=False,
    add_metadata_pre_filter=True,
    apply_new_filter=False,
    resp_type='deconv'
)
# %%

# Standard V1_nonstimd

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
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=False,
    resp_type='deconv'
)


# %% inhibited_cells

filter_steps = [
    ('response_proportion >= 0.6', lambda df: df[df['response_proportion'] >= 0.4]),

    ('peak_amp_avg >= 2', lambda df: df[df['peak_array_mean_trial'] >= 1.0]),
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
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=False,
    inhibited_cells=True,
    peak_thresh=1
)

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
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=False,
    inhibited_cells=True,
    peak_thresh=1
)
# %% inhibited single spiral
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
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=True,
    inhibited_cells=True,
    peak_thresh=1
)


area_clean, _, _, _, _, _ = process_and_summarize_single_trial_data(
    area_str='V1',
    opto_params=opto_params,
    expt_type='short_stim',
    which_neurons='non_stimd',
    signif_only=False,
    filter_steps=filter_steps,
    features_to_track=features_to_track,
    save_data=True,
    date_str='',
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=True,
    inhibited_cells=True,
    peak_thresh=1
)

# %% inhibited multi _Cell
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
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=False,
    inhibited_cells=True,
    peak_thresh=1
)


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
    return_data=True,
    add_metadata_pre_filter=True,
    apply_new_filter=False,
    inhibited_cells=True,
    peak_thresh=1
)