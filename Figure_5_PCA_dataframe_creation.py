# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 19:42:34 2025

@author: jec822
"""
#  This is only slightly modified from the pipeline in analyzeEvoked, need to do this in order to analyse the pseudo dat I generated in pseudo_trial_generation

# %%

from collections import defaultdict

def create_trial_dataframe_with_alignment(
    roi_ids, stim_ids, trial_ids, roi_keys, dff_trials, time_axes,
    peak_window_sec=[0.3, 10], align_trials=True,scramble_stim_ids=False,
    scramble_stim_ids_uncoupled=False
):
    import numpy as np
    import pandas as pd
    import analyzeEvoked2P

    if align_trials:
        dff_aligned, time_axes_aligned = analyzeEvoked2P.align_traces_from_separate_lists(
            dff_trials, time_axes
        )
        dff_trials = dff_aligned
        time_axes = time_axes_aligned

        trace_start = np.where(time_axes_aligned[0] > peak_window_sec[0])[0][0]
        trace_end = np.where(time_axes_aligned[0] > peak_window_sec[1])[0][0]
        peak_window = [trace_start, trace_end] - trace_start
    else:
        peak_window = [0, 300]
        trace_start = 100
        trace_end = None

    if scramble_stim_ids:
        # if random_seed is not None:
        #     np.random.seed(random_seed)

        # Scramble the order of dff_trials only
        dff_trials = [dff_trials[i] for i in np.random.permutation(len(dff_trials))]
        # Ensure all dff_trials are the same length — trim to shortest
        min_len = min(len(trace) for trace in dff_trials)
        dff_trials = [trace[:min_len] for trace in dff_trials]
        time_axes = [trace[:min_len] for trace in time_axes]

    if scramble_stim_ids_uncoupled:
        # Split into prestim and poststim
        prestim_parts = [trace[:trace_start].copy() for trace in dff_trials]
        poststim_parts = [trace[trace_start:].copy() for trace in dff_trials]

        # Scramble poststim across trials
        permuted_idx = np.random.permutation(len(poststim_parts))
        poststim_parts_scrambled = [poststim_parts[i] for i in permuted_idx]

        # Recombine prestim + scrambled poststim
        dff_trials = [np.concatenate([pre, post]) 
                      for pre, post in zip(prestim_parts, poststim_parts_scrambled)]

        # Ensure equal length
        min_len = min(len(trace) for trace in dff_trials)
        dff_trials = [trace[:min_len] for trace in dff_trials]
        time_axes = [trace[:min_len] for trace in time_axes]

    # Pre-index roi_keys by (roi_id, stim_id, trial_id)
    roi_key_index = defaultdict(list)
    for k in roi_keys:
        key = (k.get('roi_id'), k.get('stim_id'), k.get('trial_id'))
        roi_key_index[key].append(k)

    records = []
    
    for i, (roi_id, stim_id, trial_id, key_dict, trial_data, time_axis) in enumerate(zip(
        roi_ids, stim_ids, trial_ids, roi_keys, dff_trials, time_axes
    )):
        entry = dict(key_dict)
        entry['roi_id_extended_dataset'] = roi_id
        entry['stim_id_extended_dataset'] = stim_id
        entry['trial_id'] = trial_id
        entry['dff_trial'] = trial_data
        entry['time_axis'] = time_axis
        entry['trace_start'] = trace_start
        entry['trace_end'] = trace_end
        entry['peak_window'] = peak_window
        # Add the 0th roi_keys entry corresponding to this index:
        entry['roi_keys'] = roi_keys[i]  # just the single dict for this trial
        records.append(entry)
    
    return pd.DataFrame(records)

# %%

def baseline_sess_pca_faster(sess_key,params=params,resp_type='dff'):
    
    """
    baseline_sess_pca(expt_key,params=params,resp_type='dff',smooth_win_sec=0.5)
    performs PCA on a single baseline session, excluding stimd cells
    
    INPUTS:
        expt_key      : dict, session key
        params        : dict, analysis parameters
        resp_type     : str, 'dff' (default) or 'deconv'
    
    OUTPUTS:
        pca_results : dict, PCA results
    """
    
    start_time = time.time()
    print('     Performing baseline session PCA...')
    
    # One joined query for inclusion + stimd
    q = (
        twop_opto_analysis.TrigDffTrialAvgInclusion.proj(is_included='is_included')
        * twop_opto_analysis.TrigDffTrialAvg.proj(is_stimd='is_stimd', time_axis_sec='time_axis_sec')
        & sess_key
        & 'is_included = 1'
        & 'is_stimd = 0'
    )
    
    # Fetch all keys and the shared time axis in one go
    keys, taxis = q.fetch('KEY', 'time_axis_sec')
    
    # Restrict Dff2P to only these filtered keys
    dff = (VM['twophoton'].Dff2P & keys).fetch('dff')
    
    # Proceed
    total_frames = np.size(dff[0])
    baseline_idx = spont_timescales.get_baseline_frame_idx(sess_key, total_frames)
    frame_int    = np.diff(taxis[0])[0]
    smooth_win   = np.round(params['pca_smooth_win_sec']/frame_int).astype(int)
    
    for iCell in range(len(dff)):
        dff[iCell] = dff[iCell][:,baseline_idx].flatten()
        if smooth_win > 1:
            dff[iCell] = general_stats.moving_average(dff[iCell],num_points=smooth_win).flatten()
    
    # matrix format
    num_cells = len(dff)
    num_time    = len(dff[0])   
    dff_mat     = np.zeros((num_time,num_cells))
    dff_means   = np.zeros(num_cells)
    dff_stds    = np.zeros(num_cells)
    for iCell in range(num_cells):
        dff_mat[:,iCell] = dff[iCell]
    
        # zscore manually to put trials on same scale     
        dff_means[iCell] = np.nanmean(dff[iCell])
        dff_stds[iCell]  = np.nanstd(dff[iCell])
        dff_mat[:,iCell] = (dff_mat[:,iCell]-dff_means[iCell])/dff_stds[iCell]
    
    # run PCA
    pca_obj = PCA()
    pca_obj.fit(dff_mat)
    
    pca_results =  {'pca_obj'           : pca_obj,
                    'dff_means'         : dff_means,
                    'dff_stds'          : dff_stds,
                    'roi_keys'          : keys,
                    'params'            : deepcopy(params),
                    'resp_type'         : resp_type,
                    'smooth_win_samp'   : smooth_win,
                    'cum_var_explained' : np.cumsum(pca_obj.explained_variance_ratio_),
                    }
    
    print('     done after {:1.2f} min'.format((time.time()-start_time)/60))
    
    return pca_results


# %%


def project_trial_responses_faster(
    expt_key, 
    expt_results, 
    pca_results=None, 
    params=params, 
    resp_type='dff', 
    pca_ref=None,  
    max_number_trials=5,
    corr_method='spearman'   # NEW: 'pearson' or 'spearman'
):
    # --- decide whether to compute PCA fresh or reuse reference ---
    if pca_ref is not None:
        pca_results = pca_ref['pca_results']
    elif pca_results is None: 
        pca_results = baseline_sess_pca_faster(expt_key, params=params, resp_type=resp_type)

    keys        = pca_results['roi_keys']
    pca_obj     = pca_results['pca_obj']
    dff_means   = pca_results['dff_means']
    dff_stds    = pca_results['dff_stds']
    smooth_win  = pca_results['smooth_win_samp']
    
    trial_ids = expt_results['trial_id'].to_numpy(dtype='int64')
    trials    = expt_results['dff_trial'].to_numpy(dtype='object')
    stim_ids  = expt_results['stim_id'].to_numpy(dtype='int64')
    roi_ids   = expt_results['roi_id'].to_numpy(dtype='int64')
    taxis     = expt_results.iloc[0]['time_axis']

    # baseline / response indices
    basel_idx = np.argwhere(np.logical_and(taxis > -params['pca_basel_sec'], taxis < -0.25)).flatten()
    basel_idx = basel_idx[~np.isnan(trials[0][basel_idx])]

    first_idx = np.argwhere(np.isnan(trials[0])).flatten()[-1]+10
    last_idx  = np.argwhere(taxis <= params['pca_resp_sec']).flatten()[-1]
    resp_idx  = np.arange(first_idx, last_idx+1)
    
    stim_ids    = np.array(stim_ids).flatten()
    trial_ids   = np.array(trial_ids).flatten()
    unique_stims = np.unique(stim_ids).tolist()
    unique_rois  = np.unique(roi_ids).tolist()
    num_cells    = len(unique_rois)

    trial_projs_basel = [None]*len(unique_stims)
    trial_projs_resp  = [None]*len(unique_stims)

    for iStim, stim in enumerate(unique_stims):
        sidx = stim_ids == stim
        unique_trials = np.unique(trial_ids[sidx])[:max_number_trials].tolist()
        trial_projs_basel[iStim] = [None]*len(unique_trials)
        trial_projs_resp[iStim]  = [None]*len(unique_trials)

        for iTrial, trial in enumerate(unique_trials):
            tidx = np.logical_and(trial_ids == trial, sidx)
            trial_mat = np.zeros((len(taxis), num_cells))

            for iCell, roi in enumerate(unique_rois):
                cidx = np.argwhere(np.logical_and(tidx, roi_ids == roi)).flatten()
                if len(cidx) == 0:
                    trial_mat[:, iCell] = np.zeros(len(taxis))
                else:
                    dff = trials[cidx[0]]
                    dff = (dff - dff_means[iCell]) / dff_stds[iCell]
                    dff = general_stats.moving_average(dff, num_points=smooth_win)
                    trial_mat[:, iCell] = dff

            trial_mat = trial_mat[~np.isnan(np.median(trial_mat, axis=1))]
            trial_projs_basel[iStim][iTrial] = pca_obj.transform(trial_mat[basel_idx, :])
            trial_projs_resp[iStim][iTrial]  = pca_obj.transform(trial_mat[resp_idx, :])

    # ---- per-experiment pairwise distances (baseline & poststim) ----
    num_pcs    = params['pca_num_components']
    basel_dist = []
    resp_dist  = []
    dist_stims = []

    for iStim, stim in enumerate(unique_stims):
        num_trials = len(trial_projs_basel[iStim])
        for iTrial1 in range(num_trials):
            for iTrial2 in range(iTrial1):
                dist_b = np.linalg.norm(trial_projs_basel[iStim][iTrial1][:num_pcs] -
                                        trial_projs_basel[iStim][iTrial2][:num_pcs])
                basel_dist.append(dist_b)
                dist_r = np.linalg.norm(trial_projs_resp[iStim][iTrial1][:num_pcs] -
                                        trial_projs_resp[iStim][iTrial2][:num_pcs])
                resp_dist.append(dist_r)
                dist_stims.append(stim)

    basel_dist = np.array(basel_dist)
    resp_dist  = np.array(resp_dist)
    dist_stims = np.array(dist_stims)

    # Overall correlation
    if corr_method == 'spearman':
        corr_basel_vs_resp, p_basel_vs_resp = scipy.stats.spearmanr(basel_dist, resp_dist)
    else:
        corr_basel_vs_resp, p_basel_vs_resp = scipy.stats.pearsonr(basel_dist, resp_dist)

    # ---- correlation per stimulus pair ----
    stim_pair_rp = {}
    for stimA, stimB in combinations_with_replacement(unique_stims, 2):
        iA = unique_stims.index(stimA)
        iB = unique_stims.index(stimB)

        trialsA_basel = trial_projs_basel[iA]
        trialsB_basel = trial_projs_basel[iB]
        trialsA_resp  = trial_projs_resp[iA]
        trialsB_resp  = trial_projs_resp[iB]

        basel_pair_dists = []
        resp_pair_dists  = []

        for tA_idx, tA in enumerate(trialsA_basel):
            for tB_idx, tB in enumerate(trialsB_basel):
                if iA == iB and tA_idx >= tB_idx:
                    continue
                dist_b = np.linalg.norm(tA[:num_pcs] - tB[:num_pcs])
                dist_r = np.linalg.norm(trialsA_resp[tA_idx][:num_pcs] - trialsB_resp[tB_idx][:num_pcs])
                basel_pair_dists.append(dist_b)
                resp_pair_dists.append(dist_r)

        if corr_method == 'spearman':
            r, p = scipy.stats.spearmanr(basel_pair_dists, resp_pair_dists)
        else:
            r, p = scipy.stats.pearsonr(basel_pair_dists, resp_pair_dists)
        stim_pair_rp[(stimA, stimB)] = {'r': r, 'p': p}

    # ---- within-stim correlation (new) ----
    stim_within_rp = {}
    for iStim, stim in enumerate(unique_stims):
        basel_pair_dists = []
        resp_pair_dists  = []
        num_trials = len(trial_projs_basel[iStim])

        for iTrial1 in range(num_trials):
            for iTrial2 in range(iTrial1):
                dist_b = np.linalg.norm(trial_projs_basel[iStim][iTrial1][:num_pcs] -
                                        trial_projs_basel[iStim][iTrial2][:num_pcs])
                dist_r = np.linalg.norm(trial_projs_resp[iStim][iTrial1][:num_pcs] -
                                        trial_projs_resp[iStim][iTrial2][:num_pcs])
                basel_pair_dists.append(dist_b)
                resp_pair_dists.append(dist_r)

        if len(basel_pair_dists) > 1:
            if corr_method == 'spearman':
                r, p = scipy.stats.spearmanr(basel_pair_dists, resp_pair_dists)
            else:
                r, p = scipy.stats.pearsonr(basel_pair_dists, resp_pair_dists)
        else:
            r, p = np.nan, np.nan

        stim_within_rp[stim] = {'r': r, 'p': p}

    trial_proj_results = {
        'pca_results'       : pca_results,
        'num_pcs'           : num_pcs,
        'var_explained'     : pca_results['cum_var_explained'][num_pcs],
        'params'            : deepcopy(params),
        'trial_projs_basel' : trial_projs_basel,
        'trial_projs_resp'  : trial_projs_resp,
        'basel_dists'       : basel_dist,
        'resp_dists'        : resp_dist,
        'dist_stims'        : dist_stims,
        'p_basel_vs_resp'   : p_basel_vs_resp,
        'corr_basel_vs_resp': corr_basel_vs_resp,
        'stim_pair_rp'      : stim_pair_rp,
        'stim_within_rp'    : stim_within_rp,
        'num_stim'          : len(unique_stims)
    }

    return trial_proj_results


# %%


import numpy as np
import scipy.stats

def batch_trial_pca_faster(
    expt_keys, 
    all_expts_data, 
    params=params, 
    resp_type='dff', 
    eg_ids=None,
    prev_batch_results=None  # optional reuse of existing PCA results
):
    """
    Batch analysis of comparing baseline and post-stim PCA by calling project_trial_responses_faster.
    Collects overall distances, correlations, and per-stimulus-pair r/p values as 1D arrays.
    """
    # Filter experiments that have included cells
    valid_expt_keys = []
    for expt_key in expt_keys:
        sess_key = {'subject_fullname': expt_key['subject_fullname'],
                    'session_date'    : expt_key['session_date'],
                    'trigdff_param_set_id': params['trigdff_param_set_id_dff'],
                    'trigdff_inclusion_param_set_id': params['trigdff_inclusion_param_set_id'],
                    }
        incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & sess_key).fetch('is_included', 'KEY')
        if incl.size > 0:
            valid_expt_keys.append(expt_key)
            
    expt_keys = valid_expt_keys

    # Restrict to selected experiment groups if needed
    if eg_ids is not None:
        if not isinstance(eg_ids, list):
            eg_ids = [eg_ids]
        expt_keys = list(np.array(expt_keys)[np.array(eg_ids).astype(int)])

    # Initialize accumulators
    single_expt_results = []
    basel_dists = []
    resp_dists  = []
    dist_stims  = []
    ccs         = []
    ps          = []
    var_expl    = []
    stim_r_vals = []
    stim_p_vals = []
    stim_trial_r_vals = []
    stim_trial_p_vals = []
    cum_var_expl_per_expt = []  # NEW: store cum_var per experiment
    num_expts   = len(expt_keys)
    num_stim    = 0

    # Loop through experiments
    for ct, expt_key in enumerate(expt_keys):
        print(f"{ct+1} of {num_expts}")
        
        single_expt_data = all_expts_data[
            (all_expts_data['subject_fullname'] == expt_key['subject_fullname']) &
            (all_expts_data['session_date'] == expt_key['session_date'])
        ]
        
        # --- NEW: reuse PCA results if available ---
        if prev_batch_results is not None:
            prev_results = [
                r for r in prev_batch_results['single_expt_results']
                if (r['pca_results']['roi_keys'][0]['subject_fullname'] == expt_key['subject_fullname']) and
                   (r['pca_results']['roi_keys'][0]['session_date'] == expt_key['session_date'])
            ]
            pca_results = prev_results[0]['pca_results'] if len(prev_results) > 0 else None
        else:
            pca_results = None

        # Project data (PCA computed if pca_results is None)
        these_results = project_trial_responses_faster(
            expt_key, 
            expt_results=single_expt_data, 
            pca_results=pca_results, 
            params=params, 
            resp_type=resp_type
        )

        # --- NEW: save cum_var_explained for first num_pcs ---
        num_pcs = params['pca_num_components']
        cum_var = these_results['pca_results']['cum_var_explained']
        
        # pad with NaNs if too short
        if len(cum_var) < num_pcs*2:
            padded = np.full(num_pcs*2, np.nan)
            padded[:len(cum_var)] = cum_var
            cum_var = padded
        else:
            cum_var = cum_var[:num_pcs*2]
        
        cum_var_expl_per_expt.append(cum_var)

        # Append per-experiment results
        single_expt_results.append(these_results)
        basel_dists.extend(these_results['basel_dists'].tolist())
        resp_dists.extend(these_results['resp_dists'].tolist())
        dist_stims.extend(these_results['dist_stims'].tolist())
        ccs.append(these_results['corr_basel_vs_resp'])
        ps.append(these_results['p_basel_vs_resp'])
        var_expl.append(these_results['var_explained'])
        num_stim += these_results['num_stim']

        # Collect stim-pair r/p values as 1D arrays
        for pair, rp in these_results['stim_pair_rp'].items():
            stim_r_vals.append(rp['r'])
            stim_p_vals.append(rp['p'])

        # --- Collect trial-level r/p values ---
        if 'stim_within_rp' in these_results:
            for pair, rp in these_results['stim_within_rp'].items():
                stim_trial_r_vals.append(rp['r'])
                stim_trial_p_vals.append(rp['p'])

    # Flatten overall results
    basel_dists = np.array(basel_dists).flatten()
    resp_dists  = np.array(resp_dists).flatten()
    dist_stims  = np.array(dist_stims).flatten()
    ccs         = np.array(ccs).flatten()
    ps          = np.array(ps).flatten()
    var_expl    = np.array(var_expl).flatten()
    stim_r_vals = np.array(stim_r_vals).flatten()
    stim_p_vals = np.array(stim_p_vals).flatten()
    stim_trial_r_vals = np.array(stim_trial_r_vals).flatten()
    stim_trial_p_vals = np.array(stim_trial_p_vals).flatten()

    # Overall correlation
    corr_basel_vs_resp, p_basel_vs_resp = scipy.stats.pearsonr(basel_dists, resp_dists)
    
    trial_pca_results = {
        'single_expt_results' : single_expt_results,
        'num_pcs_used'        : params['pca_num_components'],
        'var_explained'       : var_expl,
        'cum_var_expl_per_expt': np.array(cum_var_expl_per_expt),  # NEW 2D array
        'num_sess'            : num_expts,
        'total_stimd'         : num_stim, 
        'basel_dists'         : basel_dists,
        'resp_dists'          : resp_dists,
        'dist_stims'          : dist_stims,
        'p_basel_vs_resp'     : p_basel_vs_resp,
        'corr_basel_vs_resp'  : corr_basel_vs_resp,
        'p_by_expt'           : ps,
        'corr_by_expt'        : ccs,
        'stim_pair_r'         : stim_r_vals,
        'stim_pair_p'         : stim_p_vals,
        'stim_trial_r'        : stim_trial_r_vals,
        'stim_trial_p'        : stim_trial_p_vals
    }

    return trial_pca_results

# %%

single_trial_data=single_trial_data__M2_standard_non_stimd_sig_False

data_for_pca_M2=create_trial_dataframe_with_alignment(dff_trials=single_trial_data['trig_dff_trials'],
    roi_ids=single_trial_data['roi_ids'],
    stim_ids=single_trial_data['stim_ids'],
    roi_keys=single_trial_data['roi_keys'],
    time_axes=single_trial_data['time_axis_sec'],
    trial_ids=single_trial_data['trial_ids']
)

# %% V1

single_trial_data=single_trial_data__V1_standard_non_stimd_sig_False

data_for_pca_V1=create_trial_dataframe_with_alignment(dff_trials=single_trial_data['trig_dff_trials'],
    roi_ids=single_trial_data['roi_ids'],
    stim_ids=single_trial_data['stim_ids'],
    roi_keys=single_trial_data['roi_keys'],
    time_axes=single_trial_data['time_axis_sec'],
    trial_ids=single_trial_data['trial_ids']
)

# 

# %%

resp_type = 'dff'
expt_keys = [
    {
        'subject_fullname': row['subject_fullname'],
        'session_date': row['session_date'],
        'trigdff_param_set_id': params[f'trigdff_param_set_id_{resp_type}'],
        'trigdff_inclusion_param_set_id': params['trigdff_inclusion_param_set_id']
    }
    for _, row in data_for_pca_M2[['subject_fullname', 'session_date']].drop_duplicates().iterrows()
]


m2_pca_results=batch_trial_pca_faster(expt_keys=expt_keys,all_expts_data=data_for_pca_M2,
                                      params=params, resp_type='dff', eg_ids=None,
                                       prev_batch_results=m2_pca_results
                                      )

expt_keys = [
    {
        'subject_fullname': row['subject_fullname'],
        'session_date': row['session_date'],
        'trigdff_param_set_id': params[f'trigdff_param_set_id_{resp_type}'],
        'trigdff_inclusion_param_set_id': params['trigdff_inclusion_param_set_id']
    }
    for _, row in data_for_pca_V1[['subject_fullname', 'session_date']].drop_duplicates().iterrows()
]


v1_pca_results=batch_trial_pca_faster(expt_keys,all_expts_data=data_for_pca_V1,
                                      params=params, resp_type='dff', eg_ids=None,
                                       prev_batch_results=v1_pca_results
                                      )


# %% Prints the scatter of baseline vs post stim ditances of both V1 and M2 
# doe not group by experiment

ax, results = plot_pca_dist_scatter_dual(
    area='V1',
    params=params,
    trial_pca_results_1=v1_pca_results,
    trial_pca_results_2=m2_pca_results,
    label_1='V1',
    label_2='M2',
    color_1='blue',
    color_2='orange'
)

# %%
# %% A LITTLE SLOW MAINLY BECAUSE PC EIGENVECTORS ARE  RECALCULATED EACH TIME, CAN CHANGE TO BE FASTER BUT
# 2025 09 01 ok I changed it so its much faster now, very happy with it now.

single_trial_data=single_trial_data__M2_standard_non_stimd_sig_False

# Dictionary to store results
scramble_results_dict_M2 = {}

# Repeat 50 times
for i in range(50):
    # Prepare scrambled trial data
    data_for_pca_M2_scramble = create_trial_dataframe_with_alignment(
        dff_trials=single_trial_data['trig_dff_trials'],
        roi_ids=single_trial_data['roi_ids'],
        stim_ids=single_trial_data['stim_ids'],
        roi_keys=single_trial_data['roi_keys'],
        time_axes=single_trial_data['time_axis_sec'],
        trial_ids=single_trial_data['trial_ids'],
        scramble_stim_ids=False,
        scramble_stim_ids_uncoupled=True
    )

    # Get experiment keys
    resp_type = 'dff'
    expt_keys = [
        {
            'subject_fullname': row['subject_fullname'],
            'session_date': row['session_date'],
            'trigdff_param_set_id': params[f'trigdff_param_set_id_{resp_type}'],
            'trigdff_inclusion_param_set_id': params['trigdff_inclusion_param_set_id']
        }
        for _, row in data_for_pca_M2[['subject_fullname', 'session_date']].drop_duplicates().iterrows()
    ]

    # Run PCA
    m2_pca_results_scramble = batch_trial_pca_faster(
        expt_keys=expt_keys,
        all_expts_data=data_for_pca_M2_scramble,
        params=params,
        resp_type='dff',
        eg_ids=None,
        prev_batch_results=m2_pca_results
    )

    # Store in dict
    scramble_results_dict_M2[i] = m2_pca_results_scramble


single_trial_data=single_trial_data__V1_standard_non_stimd_sig_False


scramble_results_dict_V1 = {}

# Repeat 50 times
for i in range(50):
    # Prepare scrambled trial data
    data_for_pca_V1_scramble = create_trial_dataframe_with_alignment(
        dff_trials=single_trial_data['trig_dff_trials'],
        roi_ids=single_trial_data['roi_ids'],
        stim_ids=single_trial_data['stim_ids'],
        roi_keys=single_trial_data['roi_keys'],
        time_axes=single_trial_data['time_axis_sec'],
        trial_ids=single_trial_data['trial_ids'],
        scramble_stim_ids=False,
        scramble_stim_ids_uncoupled=True
    )

    # Get experiment keys
    resp_type = 'dff'
    expt_keys = [
        {
            'subject_fullname': row['subject_fullname'],
            'session_date': row['session_date'],
            'trigdff_param_set_id': params[f'trigdff_param_set_id_{resp_type}'],
            'trigdff_inclusion_param_set_id': params['trigdff_inclusion_param_set_id']
        }
        for _, row in data_for_pca_V1[['subject_fullname', 'session_date']].drop_duplicates().iterrows()
    ]

    # Run PCA
    v1_pca_results_scramble = batch_trial_pca_faster(
        expt_keys=expt_keys,
        all_expts_data=data_for_pca_V1_scramble,
        params=params,
        resp_type='dff',
        eg_ids=None,
        prev_batch_results=v1_pca_results
    )

    # Store in dict
    scramble_results_dict_V1[i] = v1_pca_results_scramble
    


# %%
by_stim_type=True

if by_stim_type:
    corr_type='stim_trial_r'
    p_type='stim_trial_p'
else:
    corr_type='corr_by_expt'
    p_type='p_by_expt'
    
    # %%
# Collect rows from all runs
rows = []
for i in range(50):
    corr_val = scramble_results_dict_M2[i][corr_type]
    p_val = scramble_results_dict_M2[i][p_type]
    rows.append({corr_type: corr_val, p_type: p_val})

# Create DataFrame
df_corr_p_M2_scramble = pd.DataFrame(rows, columns=[corr_type, p_type])


# Collect rows from all runs
rows = []
for i in range(50):
    corr_val = scramble_results_dict_V1[i][corr_type]
    p_val = scramble_results_dict_V1[i][p_type]
    rows.append({corr_type: corr_val, p_type: p_val})

# Create DataFrame
df_corr_p_V1_scramble = pd.DataFrame(rows, columns=[corr_type, p_type])

joblib.dump(df_corr_p_M2_scramble, 'df_corr_p_M2_scramble_by_stim_type.joblib')
joblib.dump(df_corr_p_V1_scramble, 'df_corr_p_V1_scramble_by_stim_type.joblib')


# %%
import joblib

df_corr_p_M2_scramble=joblib.load('df_corr_p_M2_scramble_by_stim_type.joblib')
df_corr_p_V1_scramble=joblib.load('df_corr_p_V1_scramble_by_stim_type.joblib')

# %%

df_count_per_run_M2_scramble = pd.DataFrame({
    'num_p_below_0.05': [sum(p < 0.05 for p in p_vals) for p_vals in df_corr_p_M2_scramble['p_by_expt']]
})

df_count_per_run_V1_scramble = pd.DataFrame({
    'num_p_below_0.05': [sum(p < 0.05 for p in p_vals) for p_vals in df_corr_p_V1_scramble['p_by_expt']]
})

# %%

observed_count = 6  # from your real experiment
null_counts = df_count_per_run_M2_scramble['num_p_below_0.05']

# Empirical p-value (right-tailed test)
empirical_p = (null_counts >= observed_count).sum() / len(null_counts)

print(f"Observed: {observed_count} significant p-values")
print(f"Empirical p-value: {empirical_p:.4f}")

# %%
V1_corr_df_2d = pd.DataFrame(df_corr_p_V1_scramble['corr_by_expt'].tolist())

M2_corr_df_2d = pd.DataFrame(df_corr_p_M2_scramble['corr_by_expt'].tolist())

V1_pval_df_2d = pd.DataFrame(df_corr_p_V1_scramble['p_by_expt'].tolist())

M2_pval_df_2d = pd.DataFrame(df_corr_p_M2_scramble['p_by_expt'].tolist())
# %%

analysis_plotting_functions.plot_ecdf_comparison(df_corr_p_V1_scramble['stim_trial_r'].explode().to_numpy('float32'),
                                                 df_corr_p_M2_scramble['stim_trial_r'].explode().to_numpy('float32'),
                                                 label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                                                 line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                                                 xlabel="Tau (s)",
                                                 ylabel="",
                                                 # xticks_start=0, xticks_end=8, xticks_step=2,
                                                 # yticks_start=0, yticks_end=1, yticks_step=0.2,
                                                 # xlim=[0,8],
                                                 stat_test='auto',
                                                 figsize=[4,5],
                                                 show_normality_pvals=True,
                                                 log_x=False)

# %%

analysis_plotting_functions.plot_ecdf_comparison(v1_pca_results['corr_by_expt'],
                                                 V1_corr_df_2d.median(axis=0),
                                                 label1='V1sp data', label2='V1sp scrambled data',title='',
                                                 line_color1=params['general_params']['V1_cl'],line_color2='k',
                                                 xlabel="r value (s)",
                                                 ylabel="",
                                                  xticks_start=-0.4, xticks_end=.4, xticks_step=.2,
                                                  yticks_start=0, yticks_end=1, yticks_step=0.5,
                                                  xlim=[-0.4,0.4],
                                                 stat_test='auto',
                                                 figsize=[4,6],
                                                 show_normality_pvals=True,
                                                 log_x=False)

# %%

analysis_plotting_functions.plot_ecdf_comparison(m2_pca_results['corr_by_expt'],
                                                 M2_corr_df_2d.median(axis=0),
                                                 label1='MOs data', label2='MOs scrambled data',title='',
                                                 line_color1=params['general_params']['M2_cl'],line_color2='k',
                                                 xlabel="r value (s)",
                                                 ylabel="",
                                                  xticks_start=-0.4, xticks_end=.4, xticks_step=.2,
                                                  yticks_start=0, yticks_end=1, yticks_step=0.5,
                                                 xlim=[-0.4,0.4],
                                                 stat_test='auto',
                                                 figsize=[4,6],
                                                 show_normality_pvals=True,
                                                 log_x=False)

# %%
analysis_plotting_functions.plot_ecdf_comparison(m2_pca_results['corr_by_expt'],
                                                 v1_pca_results['corr_by_expt'],
                                                 label1='MOs data', label2='V1Sp data',title='',
                                                 line_color1=params['general_params']['M2_cl'],line_color2=params['general_params']['V1_cl'],
                                                 xlabel="r value (s)",
                                                 ylabel="",
                                                  xticks_start=-0.4, xticks_end=.4, xticks_step=.2,
                                                  yticks_start=0, yticks_end=1, yticks_step=0.5,
                                                 xlim=[-0.4,0.4],
                                                 stat_test='auto',
                                                 figsize=[4,6],
                                                 show_normality_pvals=True,
                                                 log_x=False)

# %%
V1_corr_df_2d = pd.DataFrame(df_corr_p_V1_scramble[corr_type].tolist())

M2_corr_df_2d = pd.DataFrame(df_corr_p_M2_scramble[corr_type].tolist())

V1_pval_df_2d = pd.DataFrame(df_corr_p_V1_scramble[p_type].tolist())

M2_pval_df_2d = pd.DataFrame(df_corr_p_M2_scramble[p_type].tolist())

# %% Figure 5F

analysis_plotting_functions.plot_ecdf_comparison(v1_pca_results['stim_trial_r'],
                                                 V1_corr_df_2d.median(axis=0),
                                                 label1='V1sp data', label2='V1sp scrambled data',title='',
                                                 line_color1=params['general_params']['V1_cl'],line_color2='k',
                                                 xlabel="r value (s)",
                                                 ylabel="",
                                                  xticks_start=-1, xticks_end=1, xticks_step=.5,
                                                  yticks_start=0, yticks_end=1, yticks_step=0.5,
                                                  xlim=[-1,1],
                                                 stat_test='auto',
                                                 figsize=[4,6],
                                                 show_normality_pvals=True,
                                                 log_x=False)

# %%Figure 5E

analysis_plotting_functions.plot_ecdf_comparison(m2_pca_results['stim_trial_r'],
                                                 M2_corr_df_2d.median(axis=0),
                                                 label1='MOs data', label2='MOs scrambled data',title='',
                                                 line_color1=params['general_params']['M2_cl'],line_color2='k',
                                                 xlabel="r value",
                                                 ylabel="",
                                                  xticks_start=-1, xticks_end=1, xticks_step=.5,
                                                  yticks_start=0, yticks_end=1, yticks_step=0.5,
                                                 xlim=[-1,1],
                                                 stat_test='auto',
                                                 figsize=[4,6],
                                                 show_normality_pvals=True,
                                                 log_x=False)

# %% Figure 5H

analysis_plotting_functions.plot_ecdf_comparison(v1_pca_results['stim_trial_r'],
                                                 m2_pca_results['stim_trial_r'],
                                                 label1='VISp', label2='MOs',title='',
                                                 line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                                                 xlabel="r value",
                                                 ylabel="",
                                                  xticks_start=-1, xticks_end=1, xticks_step=.5,
                                                  yticks_start=0, yticks_end=1, yticks_step=0.5,
                                                  xlim=[-1,1],
                                                  stat_test='auto',
                                                 figsize=[4,6],
                                                 show_normality_pvals=True,
                                                 log_x=False)


# %% Figure 5B

PCA_functions.plot_cum_var_explained(m2_pca_results, normalize_rows=False, figsize=(8,4), pc_colors=params['general_params']['M2_cl'],
                                     ylim=(0, .5),
                                     yticks=[0, 0.3, 0.6]
                                 )


PCA_functions.plot_cum_var_explained(v1_pca_results, normalize_rows=False, figsize=(8,4), pc_colors=params['general_params']['V1_cl'],
                                     ylim=(0, .5),
                                     yticks=[0, 0.3, 0.6]
                                 )

