# ========================================
# =============== SET UP =================
# ========================================

# %% import stuff
from utils import connect_to_dj
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pingouin as pg
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from copy import deepcopy
from schemas import spont_timescales
from schemas import twop_opto_analysis
from utils.stats import general_stats
from utils.plotting import plot_fov_heatmap
from analyzeSpont2P import params as tau_params
import analyzeSpont2P

import time

# connect to dj 
VM     = connect_to_dj.get_virtual_modules()

# %% declare default params, inherit general params from tau_params
params = {
        'random_seed'                    : 42, 
        'trigdff_param_set_id_dff'       : 4, 
        'trigdff_param_set_id_deconv'    : 5, 
        'trigdff_inclusion_param_set_id' : 2,
        'trigdff_inclusion_param_set_id_notiming' : 3, # for some analyses (e.g. inhibition), we may want to relax trough timing constraint
        'trigspeed_param_set_id'         : 1,
        'prop_resp_bins'                 : np.arange(0,.41,.01),
        'dist_bins_resp_prob'            : np.arange(30,480,50),
        'response_magnitude_bins'        : np.arange(-2.5,7.75,.25),
        'tau_bins'                       : np.arange(0,3.5,.5),
        'tau_by_time_bins'               : np.arange(0,3,.5),
        'response_time_bins'             : np.arange(0,3.1,.1),
        'fov_seq_time_bins'              : np.arange(0,12,2),
        'expression_level_type'          : 'intensity_zscore_stimdcells', # which expression level normalization to use 
        'xval_relax_timing_criteria'     : True, # set to true will select inclusion criteria that do not enforce peak timing
        'xval_recompute_timing'          : True, # set to true will recompute peak (com) for every xval iteration, false just averages existing peak times
        'xval_timing_metric'             : 'peak', # 'peak' or 'com'. peak is time of peak or trough
        'xval_num_iter'                  : 1000, # number if iters in trial_xval
        'tau_vs_opto_do_prob_by_expt'    : False, # in tau_vs_opto, do probability by experiment (vs overall)
        'tau_vs_opto_max_tau'            : 3, # max tau to include a cell in tau_vs_opto
        'prob_plot_same_scale'           : False, # in tau_vs_opto, plot all prob plots on same scale
        'pca_smooth_win_sec'             : 0.3, # window for smoothing in PCA
        'pca_num_components'             : 20, # number of PCA for trial projections
        'pca_basel_sec'                  : 2, # baseline length for pca trial analysis
        'pca_resp_sec'                   : 8, # post-stim response for pca trial analysis
        }

params['general_params'] = deepcopy(tau_params['general_params'])

# ========================================
# =============== METHODS ================
# ========================================

# ---------------
# %% get list of dj keys for sessions of a certain type
# 'standard' is single neurons with <= 10 trials & 5 spirals, 'short_stim' is single neurons <= 10 trials & < 5 spirals, 
# 'high_trial_count' is single neurons with > 10 trials, 'multi_cell' has at least one group with mutiple stim'd neurons
def get_keys_for_expt_types(area, params=params, expt_type='standard'):
    
    """
    get_keys_for_expt_types(area, params=params, expt_type='standard')
    retrieves dj keys for experiments of a certain type for a given area
    
    INPUTS:
        area      : str, 'V1' or 'M2'
        params    : dict, analysis parameters (default is params from top of this script)
        expt_type : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
    
    OUTPUTS:
        keys : list of dj keys for experiments of the desired type
    """
    
    # get primary keys for query
    mice              = params['general_params']['{}_mice'.format(area)]
    
    # get relavant keys 
    keys = list()
    for mouse in mice:
        opto_data = (twop_opto_analysis.Opto2PSummary & {'subject_fullname': mouse}).fetch(as_dict=True)
        opto_keys = (twop_opto_analysis.Opto2PSummary & {'subject_fullname': mouse}).fetch("KEY",as_dict=True)
        for ct, this_sess in enumerate(opto_data):
            has_multicell_stim = bool(np.sum(np.array(this_sess['num_cells_per_stim'])>1)>1)
            max_num_trials     = np.max(np.array(this_sess['num_trials_per_stim']))
            max_dur            = np.max(np.array(this_sess['dur_sec_per_stim']))
            
            # choose experiments that match type
            if expt_type == 'standard':
                if np.logical_and((not has_multicell_stim), np.logical_and(max_num_trials <= 10 , max_dur >= 0.2)):
                    keys.append(opto_keys[ct])
            elif expt_type == 'short_stim':
                if np.logical_and((not has_multicell_stim), np.logical_and(max_num_trials <= 10 , max_dur < 0.2)):
                    keys.append(opto_keys[ct])
            elif expt_type == 'high_trial_count':
                if np.logical_and((not has_multicell_stim) , max_num_trials > 10):
                    keys.append(opto_keys[ct])
            elif expt_type == 'multi_cell':
                if has_multicell_stim:
                    keys.append(opto_keys[ct])
            else:
                print('Unknown experiment type, doing nothing')
                return None
    
    return keys

# ---------------
# %% get proportion of significantly responding neurons for an area and experiment type
def get_prop_responding_neurons(area, params=params, expt_type='standard', resp_type='dff'):
    
    """
    get_prop_responding_neurons(area, params=params, expt_type='standard', resp_type='dff')
    retrieves proportion of significantly responding neurons for a given area and experiment type
    
    INPUTS: 
        area      : str, 'V1' or 'M2'
        params    : dict, analysis parameters (default is params from top of this script)
        expt_type : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type : str, 'dff' (default) or 'deconv'
        
    OUTPUTS:       
        summary_data : dict with summary stats
    """
    
    # get all keys
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    
    trigdff_param_set_id           = params['trigdff_param_set_id_{}'.format(resp_type)]
    trigdff_inclusion_param_set_id = params['trigdff_inclusion_param_set_id']
    
    # loop over keys and get summary stats
    prop_neurons = list()
    num_neurons  = list()
    num_stimd    = list()
    for this_key in expt_keys:
        this_key['trigdff_param_set_id']           = trigdff_param_set_id
        this_key['trigdff_inclusion_param_set_id'] = trigdff_inclusion_param_set_id
        prop, num = (twop_opto_analysis.TrigDffSummaryStats & this_key).fetch('prop_significant_rois', 'num_included_rois')
        prop_neurons.append(prop)
        num_stimd.append(len(num))
        if len(num) > 1:
            num_neurons.append(num[0])
        else:
            num_neurons.append(num)
        
    # compile resluts in a dictionary
    prop_neurons = np.array(prop_neurons).flatten()
    num_stimd    = np.array(num_stimd).flatten()
        
    summary_data = {
                    'prop_neurons_per_stim' : prop_neurons, 
                    'total_neurons_per_fov' : np.array(num_neurons).flatten(),
                    'stimd_neurons_per_fov' : num_stimd,
                    'num_unique_stimd'      : np.sum(num_stimd),
                    'num_fovs'              : len(expt_keys),
                    'prop_mean'             : np.mean(prop_neurons),
                    'prop_sem'              : np.std(prop_neurons,ddof=1) / np.sqrt(np.size(prop_neurons)-1),
                    'prop_median'           : np.median(prop_neurons),
                    'prop_iqr'              : scipy.stats.iqr(prop_neurons),
                    'response_type'         : resp_type, 
                    'experiment_type'       : expt_type, 
                    'analysis_params'       : deepcopy(params)
                    }
    
    return summary_data
     
# ---------------   
# %% plot comparison of overall proportion of significantly responding neurons
def plot_prop_response_comparison(params=params, expt_type='standard', resp_type='dff', axis_handle=None):
    
    """
    plot_prop_response_comparison(params=params, expt_type='standard', resp_type='dff', axis_handle=None)
    plots comparison of overall proportion of significantly responding neurons between V1 and M2
    
    INPUTS: 
        params     : dict, analysis parameters (default is params from top of this script)  
        expt_type  : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type  : str, 'dff' (default) or 'deconv'
        axis_handle: axis handle for plotting (optional)
        
    OUTPUTS:
        response_stats : dict with summary stats
        ax             : axis handle
    """
    
    # get data
    v1_data = get_prop_responding_neurons('V1', params=params, expt_type=expt_type, resp_type=resp_type)
    m2_data = get_prop_responding_neurons('M2', params=params, expt_type=expt_type, resp_type=resp_type)
    v1_prop = v1_data['prop_neurons_per_stim']
    m2_prop = m2_data['prop_neurons_per_stim']
    
    # compute stats
    response_stats = dict()
    response_stats['V1_summary'] = v1_data
    response_stats['M2_summary'] = m2_data
    response_stats['pval'], response_stats['test_name'] = general_stats.two_group_comparison(v1_prop, m2_prop, is_paired=False, tail="two-sided")

    # plot
    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle

    histbins     = params['prop_resp_bins']
    v1_counts, _ = np.histogram(v1_prop,bins=histbins,density=False)
    m2_counts, _ = np.histogram(m2_prop,bins=histbins,density=False)
    xaxis        = histbins[:-1]+np.diff(histbins)[0]
    ax.plot(xaxis,np.cumsum(v1_counts)/np.sum(v1_counts),color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'])
    ax.plot(xaxis,np.cumsum(m2_counts)/np.sum(m2_counts),color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'])
    ax.plot(v1_data['prop_median'],0.03,'v',color=params['general_params']['V1_cl'])
    ax.plot(m2_data['prop_median'],0.03,'v',color=params['general_params']['M2_cl'])
    ax.text(0,.8,'p = {:1.2g}'.format(response_stats['pval']),horizontalalignment='left')

    ax.set_xlabel("Prop. of responding neurons")
    ax.set_ylabel("Prop. of stim'd neurons")
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return response_stats, ax 
    

# ---------------
# %% get average opto-triiggered responses for an area and experiment type
def get_avg_trig_responses(area, params=params, expt_type='standard', resp_type='dff', eg_ids=None, signif_only=True, which_neurons='non_stimd', as_matrix=True):
    
    """
    get_avg_trig_responses(area, params=params, expt_type='standard', resp_type='dff', eg_ids=None, signif_only=True, which_neurons='non_stimd', as_matrix=True)
    retrieves opto-triggered average responses for a given area and experiment type
    
    INPUTS:
        area         : str, 'V1' or 'M2'
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type    : str, 'dff' (default) or 'deconv'
        eg_ids       : list of integers, indices of experiments to include (optional)
        signif_only  : bool, if True only include neurons that are significant (default is True)
        which_neurons: str, 'non_stimd' (default), 'all', 'stimd'
        as_matrix    : bool, if True return as matrix, if False return as list of arrays (default is True)
        
    OUTPUTS:
        summary_data : dict with summary stats and triggered averages
    """
    
    start_time      = time.time()
    print('Fetching opto-triggered averages...')
    
    # get relevant keys
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    
    # restrict to only desired rec/stim if applicable
    if eg_ids is not None:
        if isinstance(eg_ids,list) == False:
            eg_ids = [eg_ids]
        expt_keys = list(np.array(expt_keys)[np.array(eg_ids).astype(int)])
    
    # loop through keys to fetch the responses
    trigdff_param_set_id           = params['trigdff_param_set_id_{}'.format(resp_type)]
    trigdff_inclusion_param_set_id = params['trigdff_inclusion_param_set_id']
    
    avg_resps = list()
    sem_resps = list()
    t_axes    = list()
    is_stimd  = list()
    is_sig    = list()
    stim_ids  = list()
    coms      = list()
    peak_ts   = list()
    roi_keys  = list()
    num_expt  = len(expt_keys)
    
    for ct, ikey in enumerate(expt_keys):
        print('     {} of {}...'.format(ct+1,num_expt))
        this_key = {'subject_fullname' : ikey['subject_fullname'], 
                    'session_date': ikey['session_date'], 
                    'trigdff_param_set_id': trigdff_param_set_id, 
                    'trigdff_inclusion_param_set_id': trigdff_inclusion_param_set_id
                    }
        
        # for some downstream analyses we need to keep track of roi order, so separate by stim
        these_stim_ids = list((twop_opto_analysis.Opto2PSummary & this_key).fetch1('stim_ids'))
 
        for this_stim in these_stim_ids:
            this_key['stim_id'] = this_stim
            
            # do selecion at the fetching level for speed
            if which_neurons == 'stimd':
                # stimd neurons bypass inclusion criteria
                avgs, sems, ts, stimd, com, peak, sids, nkeys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('trig_dff_avg', 'trig_dff_sem', 'time_axis_sec', 'is_stimd', 'center_of_mass_sec_overall', 'time_of_peak_sec_overall', 'stim_id', 'KEY')
            else:
                sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')
                idx  = np.argwhere(np.array(incl)==1).flatten()
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx])
                
                if signif_only:
                    idx  = np.argwhere(np.array(sig)==1).flatten()
                    keys = list(np.array(keys)[idx])
                    sig  = list(np.array(sig)[idx]) 
                
                avgs, sems, ts, stimd, com, peak, sids, nkeys = (twop_opto_analysis.TrigDffTrialAvg & keys & 'is_stimd=0').fetch('trig_dff_avg', 'trig_dff_sem', 'time_axis_sec', 'is_stimd', 'center_of_mass_sec_overall', 'time_of_peak_sec_overall', 'stim_id', 'KEY')
                
                # get stimd neurons if desired (bypass inclusion because of distance criterion)
                if which_neurons == 'all':
                    avgsst, semsst, tsst, stimdst, comst, peakst, sidsst, nkeysst = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('trig_dff_avg', 'trig_dff_sem', 'time_axis_sec', 'is_stimd', 'center_of_mass_sec_overall', 'time_of_peak_sec_overall', 'stim_id', 'KEY')
                    sigst   = list(np.ones(len(avgsst)))
                    avgs    = list(avgs)
                    sems    = list(sems)
                    ts      = list(ts)
                    stimd   = list(stimd)
                    com     = list(com)
                    peak    = list(peak)
                    sids    = list(sids)
                    nkeys   = list(nkeys)
                    
                    # append to non-stimulated neurons
                    [stimd.append(st) for st in stimdst]
                    [com.append(co) for co in comst]
                    [avgs.append(avg) for avg in avgsst]
                    [sems.append(sem) for sem in semsst]
                    [ts.append(t) for t in tsst]
                    [peak.append(pt) for pt in peakst]
                    [nkeys.append(nk) for nk in nkeysst]
                    [sig.append(sg) for sg in sigst]
                    [sids.append(ss) for ss in sidsst]
                        
            # get roi keys for fetching from other tables (eg tau)
            rkeys = (VM['twophoton'].Roi2P & nkeys).fetch('KEY')

            # flatten list
            [avg_resps.append(avg) for avg in avgs]
            [sem_resps.append(sem) for sem in sems]
            [t_axes.append(t) for t in ts]
            [is_stimd.append(st) for st in stimd]
            [is_sig.append(sg) for sg in sig]
            [coms.append(co) for co in com]
            [peak_ts.append(pt) for pt in peak]
            [stim_ids.append(ss) for ss in sids]
            [roi_keys.append(rr) for rr in rkeys]
            
    is_stimd = np.array(is_stimd).flatten()
    is_sig   = np.array(is_sig).flatten()
    peak_ts  = np.array(peak_ts).flatten()
    coms     = np.array(coms).flatten()
    stim_ids = np.array(stim_ids).flatten()

    # interpolate to put everyone on the exact same time axis (small diffs in frame rate are possible)
    # start by aligning all time axes to zero and taking the mode of each bin
    nt_pre  = list()
    nt_post = list()
    fdur    = list()
    for taxis in t_axes:
        nt_pre.append(np.sum(taxis<0))
        nt_post.append(np.sum(taxis>=0))
        fdur.append(np.diff(taxis)[0])
    
    # base time axis making sure to include t = 0
    fdur, _    = scipy.stats.mode(np.array(fdur).flatten())
    nt_pre, _  = scipy.stats.mode(np.array(nt_pre).flatten()) 
    nt_post, _ = scipy.stats.mode(np.array(nt_post).flatten())   
    pre_t      = np.arange(-nt_pre*fdur,0,fdur)
    post_t     = np.arange(0,nt_post*fdur,fdur) 
    base_taxis = np.concatenate((pre_t,post_t)) 
    
    # convert all axes to base (mostly expected to be unchanged)
    for iResp in range(len(t_axes)):
        this_avg = deepcopy(avg_resps[iResp])
        this_avg[this_avg < -10] = np.nan
        this_sem = deepcopy(sem_resps[iResp])
        this_sem[this_sem < -10] = np.nan
        avg_resps[iResp] = np.interp(base_taxis,t_axes[iResp],this_avg)  
        sem_resps[iResp] = np.interp(base_taxis,t_axes[iResp],this_sem)    
        
    # convert from list to matrix if desired    
    if as_matrix:
        avgs = np.zeros((len(t_axes),len(base_taxis)))
        sems = np.zeros((len(t_axes),len(base_taxis)))
        for iResp in range(len(t_axes)):
            avgs[iResp,:] = avg_resps[iResp]
            sems[iResp,:] = sem_resps[iResp]
            
        # make sure NaN frames for shuttered pmt are the same
        nan_idx1    = np.argwhere(np.isnan(np.sum(avgs,axis=0))).flatten()
        if np.size(nan_idx1) > 0:
            nan_idx     = np.zeros(np.size(nan_idx1)+1)
            nan_idx[0]  = nan_idx1[0]-1
            nan_idx[1:] = nan_idx1
            avgs[:,nan_idx.astype(int)] = np.nan
            sems[:,nan_idx.astype(int)] = np.nan
    else:
        avgs = avg_resps
        sems = sem_resps
        
    # collect summary data   
    summary_data = {
                    'trig_dff_avgs'  : avgs, 
                    'trig_dff_sems'  : sems,
                    'time_axis_sec'  : base_taxis,
                    'num_responding' : len(avg_resps),
                    'is_stimd'       : is_stimd,
                    'is_sig'         : is_sig,
                    'peak_times_sec' : peak_ts,
                    'com_sec'        : coms,
                    'stim_ids'       : stim_ids,
                    'response_type'  : resp_type, 
                    'experiment_type': expt_type, 
                    'which_neurons'  : which_neurons,
                    'roi_keys'       : roi_keys,
                    'analysis_params': deepcopy(params)
                    }
    
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    return summary_data

# ---------------
# %% get response stats (peak, dist from stim etc) for an area and experiment type
def get_full_resp_stats(area, params=params, expt_type='standard', resp_type='dff', eg_ids=None, which_neurons='non_stimd'):
    
    """
    get_full_resp_stats(area, params=params, expt_type='standard', resp_type='dff', eg_ids=None, which_neurons='non_stimd')
    retrieves opto-triggered response stats for a given area and experiment type
    
    INPUTS:
        area         : str, 'V1' or 'M2'
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type    : str, 'dff' (default) or 'deconv'
        eg_ids       : list of integers, indices of experiments to include (optional)
        which_neurons: str, 'non_stimd' (default), 'all', 'stimd'
        
    OUTPUTS:
        summary_data : dict with summary stats 
    """
    
    start_time      = time.time()
    print('Fetching opto-triggered response stats...')
    
    # get relevant keys
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    
    # restrict to only desired rec/stim if applicable
    if eg_ids is not None:
        if isinstance(eg_ids,list) == False:
            eg_ids = [eg_ids]
        expt_keys = list(np.array(expt_keys)[np.array(eg_ids).astype(int)])
    
    # loop through keys to fetch the responses
    trigdff_param_set_id           = params['trigdff_param_set_id_{}'.format(resp_type)]
    trigdff_inclusion_param_set_id = params['trigdff_inclusion_param_set_id']
    
    is_stimd   = list()
    is_sig     = list()
    coms       = list()
    maxmins_ts = list()
    maxmins    = list()
    dists      = list()
    num_expt   = len(expt_keys)
    dist_by_expt   = list()
    is_sig_by_expt = list()
    n_by_expt      = list()
    n_sig_by_expt  = list()
    prop_vs_total  = list()
    prop_of_sig    = list()
    roi_keys       = list()
    stim_ids       = list()
    dist_bins      = params['dist_bins_resp_prob']
    num_bins       = len(dist_bins)-1
    
    for ct, ikey in enumerate(expt_keys):
        print('     {} of {}...'.format(ct+1,num_expt))
        this_key = {'subject_fullname' : ikey['subject_fullname'], 
                    'session_date': ikey['session_date'], 
                    'trigdff_param_set_id': trigdff_param_set_id, 
                    'trigdff_inclusion_param_set_id': trigdff_inclusion_param_set_id
                    }
        
        # for some downstream analyses we need to keep track of roi order, so separate by stim
        these_stim_ids = list((twop_opto_analysis.Opto2PSummary & this_key).fetch1('stim_ids'))
 
        for this_stim in these_stim_ids:
            this_key['stim_id'] = this_stim
            
            # do selecion at the fetching level for speed
            if which_neurons == 'stimd':
                # stimd neurons bypass inclusion criteria
                stimd, com, maxmin, peakt, trought, dist, sid, nkeys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')
                n    = len(stimd)
                sig  = list(np.ones(n))
            else:          
                # get included non-stimulated neurons first 
                sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')
                idx  = np.argwhere(np.array(incl)==1).flatten()
                n    = np.size(idx)
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx])

                stimd, com, maxmin, peakt, trought, dist, sid, nkeys = (twop_opto_analysis.TrigDffTrialAvg & keys & 'is_stimd=0').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')
                
                # get stimd neurons if desired (bypass inclusion because of distance criterion)
                if which_neurons == 'all':
                    stimdst, comst, maxminst, peaktst, troughtst, distst, sidst, nkeysst = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')
                    sigs  = list(np.ones(len(stimdst)))
                    
                    stimd   = list(stimd)
                    com     = list(com)
                    maxmin  = list(maxmin)
                    peakt   = list(peakt)
                    trought = list(trought)
                    dist    = list(dist)
                    sid     = list(sid)
                    nkeys   = list(nkeys)
                    
                    # append to non-stimulated neurons
                    [stimd.append(st) for st in stimdst]
                    [com.append(co) for co in comst]
                    [maxmin.append(mm) for mm in maxminst]
                    [peakt.append(pt) for pt in peaktst]
                    [trought.append(tr) for tr in troughtst]
                    [dist.append(ds) for ds in distst]
                    [sid.append(ss) for ss in sidst]
                    [nkeys.append(nk) for nk in nkeysst]
                    [sig.append(sg) for sg in sigs]
                    
            # pick trough or peak time, whichever is higher magnitude
            maxmin_t = list()
            for iNeuron in range(len(trought)):
                if maxmin[iNeuron] < 0:
                    maxmin_t.append(trought[iNeuron])
                else:
                    maxmin_t.append(peakt[iNeuron])
                    
            # get roi keys for fetching from other tables (eg tau)
            rkeys = (VM['twophoton'].Roi2P & nkeys).fetch('KEY')

            # flatten list
            [is_stimd.append(st) for st in stimd]
            [is_sig.append(sg) for sg in sig]
            [coms.append(co) for co in com]
            [maxmins.append(mm) for mm in maxmin]
            [maxmins_ts.append(mmt) for mmt in maxmin_t]
            [dists.append(ds) for ds in dist]
            [roi_keys.append(rk) for rk in rkeys]
            [stim_ids.append(ss) for ss in sid]

            # response prob by dist per experiment
            dist        = np.array(dist).flatten()
            sig         = np.array(sig).flatten()
            unique_sids = np.unique(sid)
            n           = n / len(unique_sids)
            vs_total    = np.zeros(num_bins)
            vs_sig      = np.zeros(num_bins)
            for this_id in unique_sids:
                this_dist = dist[sid == this_id]
                this_sig  = sig[sid == this_id]
                dist_by_expt.append(this_dist)
                is_sig_by_expt.append(this_sig)
                n_sig_by_expt.append(np.sum(this_sig==1))
                n_by_expt.append(n)
                for iBin in range(num_bins):
                    idx    = np.logical_and(this_dist > dist_bins[iBin], this_dist <= dist_bins[iBin+1])
                    this_n = np.sum(this_sig[idx==1])
                    vs_total[iBin] = this_n / n
                    if n_sig_by_expt[-1] == 0:
                        vs_sig[iBin] = 0
                    else:
                        vs_sig[iBin] = this_n / n_sig_by_expt[-1]
                    
                prop_vs_total.append(vs_total)
                prop_of_sig.append(vs_sig)
    
    # collect summary data   
    summary_data = {
                    'num_total_neurons'     : np.sum(np.array(n_by_expt).flatten()),
                    'num_responding'        : np.sum(np.array(n_sig_by_expt).flatten()),
                    'num_experiments'       : len(prop_vs_total),
                    'num_sig_by_expt'       : np.array(n_sig_by_expt).flatten(),
                    'is_stimd'              : np.array(is_stimd).flatten(),
                    'stim_ids'              : np.array(stim_ids).flatten(),
                    'is_sig'                : np.array(is_sig).flatten(),
                    'max_or_min_times_sec'  : np.array(maxmins_ts).flatten(),
                    'max_or_min_vals'       : np.array(maxmins).flatten(),
                    'com_sec'               : np.array(coms).flatten(),
                    'dist_from_stim_um'     : np.array(dists).flatten(),
                    'dist_axis'             : dist_bins[:-1]+np.diff(dist_bins)[0]/2,
                    'prop_by_dist_vs_total' : prop_vs_total,
                    'prop_by_dist_of_sig'   : prop_of_sig,
                    'response_type'         : resp_type, 
                    'experiment_type'       : expt_type, 
                    'which_neurons'         : which_neurons,
                    'roi_keys'              : roi_keys,
                    'analysis_params'       : deepcopy(params)
                    }
    
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    return summary_data

# ---------------
# %% compare general responses stats between areas
def compare_response_stats(params=params, expt_type='standard', resp_type='dff', which_neurons='non_stimd', v1_data=None, m2_data=None, signif_only=True):
    
    """
    compare_response_stats(params=params, expt_type='standard', resp_type='dff', which_neurons='non_stimd', v1_data=None, m2_data=None)
    compares general response stats between V1 and M2
    
    INPUTS: 
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type    : str, 'dff' (default) or 'deconv'
        which_neurons: str, 'non_stimd' (default), 'all', 'stimd'
        v1_data      : dict with V1 response stats, output of get_full_resp_stats (optional, if not provided will call that method)
        m2_data      : dict with M2 response stats, get_full_resp_stats (optional, if not provided will call that method)
        signif_only  : bool, if True only include neurons that are significant (default is True)
        
    OUTPUTS:
        response_stats : dict with summary stats
    """
    
    # get data
    if v1_data is None:
        v1_data = get_full_resp_stats('V1', params=params, expt_type=expt_type, resp_type=resp_type, which_neurons=which_neurons)
    if m2_data is None:
        m2_data = get_full_resp_stats('M2', params=params, expt_type=expt_type, resp_type=resp_type, which_neurons=which_neurons)

    # compute stats
    response_stats = dict()
    response_stats['V1_stats'] = v1_data
    response_stats['M2_stats'] = m2_data
    
    # two-group comparisons
    if signif_only:
        idx_v1 = np.argwhere(v1_data['is_sig']==1).flatten()
        idx_m2 = np.argwhere(m2_data['is_sig']==1).flatten()
    else:
        idx_v1 = np.arange(len(v1_data['is_sig']))
        idx_m2 = np.arange(len(m2_data['is_sig']))
    response_stats['response_magnitude_pval'], response_stats['response_magnitude_test_name'] = \
        general_stats.two_group_comparison(v1_data['max_or_min_vals'][idx_v1], m2_data['max_or_min_vals'][idx_m2], is_paired=False, tail="two-sided")
    response_stats['response_time_pval'], response_stats['response_time_test_name'] = \
        general_stats.two_group_comparison(v1_data['max_or_min_times_sec'][idx_v1], m2_data['max_or_min_times_sec'][idx_m2], is_paired=False, tail="two-sided")
    response_stats['response_com_pval'], response_stats['response_com_test_name'] = \
        general_stats.two_group_comparison(v1_data['com_sec'][idx_v1], m2_data['com_sec'][idx_m2], is_paired=False, tail="two-sided")

    # two-way RM ANOVAs for the space-dependent metrics 
    # make it flattened lists for dataframe conversion first
    dists      = list()
    areas      = list()
    prop_total = list()
    prop_sig   = list()
    dist_vals  = list(v1_data['dist_axis'])
    num_bins   = len(dist_vals)
    for iEx in range(v1_data['num_experiments']):
        [prop_total.append(ii) for ii in list(v1_data['prop_by_dist_vs_total'][iEx])]
        [prop_sig.append(ii) for ii in list(v1_data['prop_by_dist_of_sig'][iEx])]
        [dists.append(ii) for ii in list(dist_vals)]
        [areas.append(ii) for ii in ['V1']*num_bins]
    for iEx in range(m2_data['num_experiments']):
        [prop_total.append(ii) for ii in list(m2_data['prop_by_dist_vs_total'][iEx])]
        [prop_sig.append(ii) for ii in list(m2_data['prop_by_dist_of_sig'][iEx])]
        [dists.append(ii) for ii in list(dist_vals)]
        [areas.append(ii) for ii in ['M2']*num_bins]
        
    df         = pd.DataFrame({'area': areas, 
                               'dist': dists, 
                               'prop_vs_total': prop_total, 
                               'prop_of_sig'  : prop_sig
                               })

    response_stats['anova_prop_by_dist_vs_total'] = pg.anova(data=df, dv='prop_vs_total', between=['area', 'dist'])
    response_stats['anova_prop_by_dist_of_sig']   = pg.anova(data=df, dv='prop_of_sig', between=['area', 'dist'])

    return response_stats

# ---------------
# %% plot comparison of different response stats between areas
def plot_response_stats_comparison(params=params, expt_type='standard', resp_type='dff', which_neurons='non_stimd', response_stats=None, axis_handle=None, plot_what='response_magnitude', signif_only=True, overlay_non_sig=False):
    
    """
    plot_response_stats_comparison(params=params, expt_type='standard', resp_type='dff', which_neurons='non_stimd', response_stats=None, axis_handle=None, plot_what='response_magnitude')
    plots comparison of response stats between V1 and M2
    
    INPUTS:
        params         : dict, analysis parameters (default is params from top of this script)
        expt_type      : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type      : str, 'dff' (default) or 'deconv'
        which_neurons  : str, 'non_stimd' (default), 'all', 'stimd'
        response_stats : dict with response stats, output of compare_response_stats (optional, if not provided will call that method)
        axis_handle    : axis handle for plotting (optional)
        plot_what      : str, 'response_magnitude' (default), 'response_time', 'prop_by_dist_vs_total' (i.e. overall prop.), 'prop_by_dist_of_sig' (i.e. prop. of significant)
        signif_only    : bool, if True only include neurons that are significant (default is True)
        overlay_non_sig: bool, if True overlay non-significant neurons (default is False)
        
    OUTPUTS:
        response_stats : dict with summary stats
        ax             : axis handle
    """
    
    # call this one dedicated function if it's just overall proportion (there for historical reasons)
    if plot_what == 'response_probability':
        response_stats, ax = plot_prop_response_comparison(params=params, expt_type=expt_type, resp_type=resp_type, axis_handle=axis_handle)
        return response_stats, ax
    
    # get data
    if response_stats is None:
        response_stats = compare_response_stats(params=params, expt_type=expt_type, resp_type=resp_type, which_neurons=which_neurons,signif_only=signif_only)
            
    # indices for significant neurons
    if signif_only:
        idx_v1 = np.argwhere(response_stats['V1_stats']['is_sig']==1).flatten()
        idx_m2 = np.argwhere(response_stats['M2_stats']['is_sig']==1).flatten()
    else:
        idx_v1 = np.arange(len(response_stats['V1_stats']['is_sig']))
        idx_m2 = np.arange(len(response_stats['M2_stats']['is_sig']))
        
    # isolate desired variables    
    if plot_what == 'response_magnitude':
        v1_data  = response_stats['V1_stats']['max_or_min_vals'][idx_v1]
        m2_data  = response_stats['M2_stats']['max_or_min_vals'][idx_m2]
        full_v1  = response_stats['V1_stats']['max_or_min_vals']
        full_m2  = response_stats['M2_stats']['max_or_min_vals']
        pval     = response_stats[plot_what+'_pval']
        stats    = None
        histbins = params[plot_what+'_bins']
        xlbl     = 'Response magnitude (z-score)'
        ylbl     = 'Prop. of responding neurons'
        
    elif plot_what == 'response_time':
        v1_data  = response_stats['V1_stats']['max_or_min_times_sec'][idx_v1]
        m2_data  = response_stats['M2_stats']['max_or_min_times_sec'][idx_m2]
        full_v1  = response_stats['V1_stats']['max_or_min_times_sec']
        full_m2  = response_stats['M2_stats']['max_or_min_times_sec']
        pval     = response_stats[plot_what+'_pval']
        stats    = None
        histbins = params[plot_what+'_bins']
        xlbl     = 'Response peak (trough) time (sec)'
        ylbl     = 'Prop. of responding neurons'
        
    elif plot_what == 'prop_by_dist_vs_total' or plot_what == 'prop_by_dist_of_sig':
        v1_data  = response_stats['V1_stats'][plot_what]
        m2_data  = response_stats['M2_stats'][plot_what]
        pval     = None
        stats    = response_stats['anova_'+plot_what]
        histbins = None
        xlbl     = 'Dist. from stimd ($\\mu$m)'
        ylbl     = 'Prop. responding neurons'
        if plot_what == 'prop_by_dist_of_sig':
            ylbl     = 'Prop. sig. responding neurons'
            
    else:
        print('unknown plot_what, plotting nothing')
        return response_stats, None
    
    # plot
    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle

    if plot_what == 'prop_by_dist_vs_total' or plot_what == 'prop_by_dist_of_sig':
        # by dist plots
        xaxis   = response_stats['V1_stats']['dist_axis']
        resp_v1 = np.zeros((len(v1_data),np.size(xaxis)))
        resp_m2 = np.zeros((len(m2_data),np.size(xaxis)))
        for idx, this_resp in enumerate(v1_data):
            resp_v1[idx,:] = this_resp
        for idx, this_resp in enumerate(m2_data):
            resp_m2[idx,:] = this_resp
            
        v1_mean = np.nanmean(resp_v1,axis=0)
        v1_sem  = np.nanstd(resp_v1,axis=0) / np.sqrt(len(v1_data)-1)
        m2_mean = np.nanmean(resp_m2,axis=0)
        m2_sem  = np.nanstd(resp_m2,axis=0) / np.sqrt(len(m2_data)-1)
        
        pval_area     = stats['p-unc'].loc[0]
        pval_dist     = stats['p-unc'].loc[1]
        pval_interact = stats['p-unc'].loc[2]
        
        if plot_what == 'prop_by_dist_vs_total':
            texty = 0.025
        else:
            texty = 0.35
        
        ax.errorbar(x=xaxis,y=v1_mean,yerr=v1_sem,color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'],marker='.')
        ax.errorbar(x=xaxis,y=m2_mean,yerr=m2_sem,color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'],marker='.')
        ax.text(200,texty,'p(dist) = {:1.2g}'.format(pval_dist),horizontalalignment='left')
        ax.text(200,texty*.95,'p(area) = {:1.2g}'.format(pval_area),horizontalalignment='left')
        ax.text(200,texty*.9,'p(area*dist) = {:1.2g}'.format(pval_interact),horizontalalignment='left')
        
    else:
        # cumulative histograms
        if overlay_non_sig:
            v1_counts, _ = np.histogram(full_v1,bins=histbins,density=False)
            m2_counts, _ = np.histogram(full_m2,bins=histbins,density=False)
            xaxis        = histbins[:-1]+np.diff(histbins)[0]
            ax.plot(xaxis,np.cumsum(v1_counts)/np.sum(v1_counts),color=params['general_params']['V1_sh'],label=params['general_params']['V1_lbl']+'(all)')
            ax.plot(xaxis,np.cumsum(m2_counts)/np.sum(m2_counts),color=params['general_params']['M2_sh'],label=params['general_params']['M2_lbl']+'(all)')
            ax.plot(np.median(full_v1),0.03,'v',color=params['general_params']['V1_sh'])
            ax.plot(np.median(full_m2),0.03,'v',color=params['general_params']['M2_sh'])
            
        v1_counts, _ = np.histogram(v1_data,bins=histbins,density=False)
        m2_counts, _ = np.histogram(m2_data,bins=histbins,density=False)
        xaxis        = histbins[:-1]+np.diff(histbins)[0]
        ax.plot(xaxis,np.cumsum(v1_counts)/np.sum(v1_counts),color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'])
        ax.plot(xaxis,np.cumsum(m2_counts)/np.sum(m2_counts),color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'])
        ax.plot(np.median(v1_data),0.03,'v',color=params['general_params']['V1_cl'])
        ax.plot(np.median(m2_data),0.03,'v',color=params['general_params']['M2_cl'])
        ax.text(xaxis[0]+.02,.8,'p = {:1.2g}'.format(pval),horizontalalignment='left')

    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return response_stats, ax 

# ---------------
# %% get opto-triggered running  data
def get_trig_speed(area, params=params, expt_type='standard', as_matrix=True):
    
    """
    get_trig_speed(area, params=params, expt_type='standard', as_matrix=True)
    retrieves opto-triggered running data for a given area and experiment type
    
    INPUTS:
        area         : str, 'V1' or 'M2'
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        as_matrix    : bool, if True return as matrix, if False return as list of arrays (default is True)
        
    OUTPUTS:
        trig_speed_data : dict with summary stats and triggered averages
    """
    start_time      = time.time()
    print('Fetching opto-triggered running...')
    
    # get relevant keys
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    full_keys = list()
    for key in expt_keys:
        key['trigspeed_param_set_id'] = params['trigspeed_param_set_id']
        full_keys.append(key)
    
    # loop for trig avgs
    avg_speed = list()
    t_axes    = list()
    num_expt  = len(expt_keys)
    for ct, key in enumerate(full_keys):
        print('     {} of {}...'.format(ct+1,num_expt))
        speed, taxis = (twop_opto_analysis.TrigRunningTrialAvg & key).fetch('trig_running_avg', 'time_axis_sec')
        [avg_speed.append(istim) for istim in speed]
        [t_axes.append(istim) for istim in taxis]
        
    # interpolate to put every expt o exactly the same time axis
    nt_pre  = list()
    nt_post = list()
    fdur    = list()
    for taxis in t_axes:
        nt_pre.append(np.size(taxis[taxis<0]))
        nt_post.append(np.size(taxis[taxis>=0]))
        fdur.append(np.diff(taxis)[0])
    
    # base time axis making sure to include t = 0
    fdur, _    = scipy.stats.mode(np.array(fdur).flatten())
    nt_pre, _  = scipy.stats.mode(np.array(nt_pre).flatten()) 
    nt_post, _ = scipy.stats.mode(np.array(nt_post).flatten())   
    pre_t      = np.arange(-nt_pre*fdur,0,fdur)
    post_t     = np.arange(0,nt_post*fdur,fdur) 
    base_taxis = np.concatenate((pre_t,post_t)) 
    
    # convert all axes to base (mostly expected to be unchanged), z score
    for iResp in range(len(t_axes)):
        avg_speed[iResp] = np.interp(base_taxis,t_axes[iResp],avg_speed[iResp])  
        
    # convert from list to matrix if desired    
    if as_matrix:
        avgs   = np.zeros((len(t_axes),len(base_taxis)))
        for iResp in range(len(t_axes)):
            avgs[iResp,:]   = avg_speed[iResp]
    else:
        avgs   = avg_speed
               
    trig_speed_data = {
                    'num_experiments'   : len(t_axes),
                    'time_axis_sec'     : base_taxis,
                    'trig_speed'        : avgs,
                    'trig_speed_mean'   : np.nanmean(avgs,axis=0),
                    'trig_speed_median' : np.nanmedian(avgs,axis=0),
                    'trig_speed_sem'    : np.nanstd(avgs,axis=0)/np.sqrt(len(t_axes)-1),
                    'trig_speed_iqr'    : scipy.stats.iqr(avgs,axis=0),
                    'experiment_type'   : expt_type, 
                    'analysis_params'   : deepcopy(params)
                    }
                
    return trig_speed_data

# -----------------------
# %% get opto-triggered running  data
def plot_trig_speed(params=params, expt_type='standard', v1_data=None, m2_data=None, axis_handle=None):
    
    """
    plot_trig_speed(params=params, expt_type='standard', v1_data=None, m2_data=None, axis_handle=None)
    plots opto-triggered running data for V1 and M2
    
    INPUTS:
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        v1_data      : dict with V1 running stats, output of get_trig_speed (optional, if not provided will call that method)
        m2_data      : dict with M2 running stats, get_trig_speed (optional, if not provided will call that method)
        axis_handle  : axis handle for plotting (optional)
        
    OUTPUTS:
        trig_speed_stats : dict with summary stats
        ax               : axis handle
    """
    
    # get data if necessary
    if v1_data is None:
        v1_data = get_trig_speed('V1', params=params, expt_type=expt_type)
    if m2_data is None:
        m2_data = get_trig_speed('M2', params=params, expt_type=expt_type)
    
    # stats
    trig_speed_stats = dict()
    trig_speed_stats['V1_summary'] = v1_data
    trig_speed_stats['M2_summary'] = m2_data
    
    # test for avg pre vs post running
    xaxis_v1 = v1_data['time_axis_sec']
    xaxis_m2 = m2_data['time_axis_sec']
    v1_pre  = np.nanmedian(v1_data['trig_speed'][:,xaxis_v1<0],axis=1).flatten()
    v1_post = np.nanmedian(v1_data['trig_speed'][:,np.logical_and(xaxis_v1>0, xaxis_v1<10)],axis=1).flatten()
    m2_pre  = np.nanmedian(m2_data['trig_speed'][:,xaxis_m2<0],axis=1).flatten()
    m2_post = np.nanmedian(m2_data['trig_speed'][:,np.logical_and(xaxis_m2>0, xaxis_m2<10)],axis=1).flatten()
    trig_speed_stats['pval_v1'], trig_speed_stats['test_name_v1'] = \
        general_stats.two_group_comparison(v1_pre, v1_post, is_paired=True, tail="two-sided")
    trig_speed_stats['pval_m2'], trig_speed_stats['test_name_m2'] = \
        general_stats.two_group_comparison(m2_pre, m2_post, is_paired=True, tail="two-sided")
    trig_speed_stats['pval_m2_vs_v1'], trig_speed_stats['test_name_m2_vs_v1'] = \
        general_stats.two_group_comparison(m2_post - m2_pre, v1_post - v1_pre, is_paired=False, tail="two-sided")
        
    # plot
    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle
        
    mean_v1  = v1_data['trig_speed_median']
    mean_m2  = m2_data['trig_speed_median']
    sem_v1   = v1_data['trig_speed_iqr']
    sem_m2   = m2_data['trig_speed_iqr']
    
    ax.fill_between(xaxis_v1,mean_v1-sem_v1,mean_v1+sem_v1,color=params['general_params']['V1_sh'])
    ax.plot(xaxis_v1,mean_v1,'-',color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'])
    ax.fill_between(xaxis_m2,mean_m2-sem_m2,mean_m2+sem_m2,color=params['general_params']['M2_sh'])
    ax.plot(xaxis_m2,mean_m2,'-',color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'])
    yl = ax.get_ylim()
    ax.plot([0,0],yl,'--',color=[.8,.8,.8])
    ax.text(-2.5,yl[0]*.9,'p({}) = {:1.2g}'.format(params['general_params']['V1_lbl'],trig_speed_stats['pval_v1']),horizontalalignment='left')
    ax.text(-2.5,yl[0]*.82,'p({}) = {:1.2g}'.format(params['general_params']['M2_lbl'],trig_speed_stats['pval_m2']),horizontalalignment='left')
    
    ax.set_xlabel('Time from stim (sec)')
    ax.set_ylabel('Running speed (z-score)')
    ax.set_xlim((-3,10))
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return trig_speed_stats, ax     

# ---------------
# %% get opsin expression for stimd cells and correlate with response magnitude
def get_opsin_expression_vs_response(area, params=params, expt_type='standard'):
    
    """
    get_opsin_expression_vs_response(area, params=params, expt_type='standard')
    retrieves opsin expression levels for stim'd cells and correlates with response magnitude
    
    INPUTS:
        area         : str, 'V1' or 'M2'
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        
    OUTPUTS:
        expression_data : dict with summary data
    """
    
    start_time      = time.time()
    print('Running expression level analysis...')
    
    # get relevant keys
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    num_expt  = len(expt_keys)
    
    expr_lvls    = list()
    stimd_resp   = list()
    network_resp = list()
    # for each experiment, get opsin expression levels for every stim'd neuron
    for expt_ct, key in enumerate(expt_keys):
        print('     {} of {}...'.format(expt_ct+1,num_expt))
        
        ids_list, stim_ids    = (twop_opto_analysis.Opto2PSummary & key).fetch1('roi_id_per_stim','stim_ids')
        del key['scan_number']
        seg_key      = twop_opto_analysis.get_single_segmentation_key(key)
        key['trigdff_param_set_id']           = params['trigdff_param_set_id_{}'.format('dff')]
        key['trigdff_inclusion_param_set_id'] = params['trigdff_inclusion_param_set_id']
                    
        # because some experiments have multiple simulatenous neurons, we'll just take the average
        # for each stim_id group. Will be equivalent to single-neuron  in most cases
        for ct, roi_ids in enumerate(ids_list):
            key['stim_id'] = int(stim_ids[ct])
            roi_ids = [int(rid[0][0]) for rid in roi_ids]
            expr    = list()
            st_resp = list()
            for rid in roi_ids:
                expr.append((twop_opto_analysis.OpsinExpression & seg_key & {'roi_id': rid}).fetch1(params['expression_level_type']))
                # get peak abs response of stim'd ensample
                st_resp.append((twop_opto_analysis.TrigDffTrialAvg & key & {'roi_id': rid}).fetch1('max_or_min_dff'))
                
            expr_lvls.append(np.median(np.array(expr)))
            stimd_resp.append(np.median(np.abs(np.array(st_resp))))
            
            # get peak abs response of stim'd ensample, and of the rest of significantly responding cells
            nonstimd_keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & key & 'is_included=1' & 'is_significant=1').fetch('KEY')
            if len(nonstimd_keys) > 0:
                nt_resp       = (twop_opto_analysis.TrigDffTrialAvg &  nonstimd_keys).fetch('max_or_min_dff')
                network_resp.append(np.nanmedian(np.abs(np.array(nt_resp))))
            else:
                network_resp.append(np.nan)
            
    # condition, collect data and run stats
    expr_lvls    = np.array(expr_lvls).flatten()
    stimd_resp   = np.array(stimd_resp).flatten()
    network_resp = np.array(network_resp).flatten()
    corr_stimd, p_stimd = scipy.stats.pearsonr(expr_lvls[~np.isnan(stimd_resp)],stimd_resp[~np.isnan(stimd_resp)])
    corr_netw, p_netw   = scipy.stats.pearsonr(expr_lvls[~np.isnan(network_resp)],network_resp[~np.isnan(network_resp)])
    
    expression_data = {
                    'num_stimd_groups'               : np.size(expr_lvls),
                    'expression_level_type'          : params['expression_level_type'],
                    'expression_levels'              : expr_lvls,
                    'abs_resp_stimd'                 : stimd_resp,
                    'abs_resp_non_stimd'             : network_resp,
                    'cc_express_vs_stimd_resp'       : corr_stimd,
                    'pval_express_vs_stimd_resp'     : p_stimd,
                    'cc_express_vs_non_stimd_resp'   : corr_netw,
                    'pval_express_vs_non_stimd_resp' : p_netw,
                    'experiment_type'                : expt_type, 
                    'analysis_params'                : deepcopy(params)
                    }
    
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    return expression_data

# ---------------
# %% plot correlation between opsin expression for stimd cells and response magnitude
def plot_opsin_expression_vs_response(params=params, expt_type='standard', resp_type='dff', v1_data=None, m2_data=None, plot_what='stimd', axis_handle=None):
    
    """
    plot_opsin_expression_vs_response(params=params, expt_type='standard', resp_type='dff', v1_data=None, m2_data=None, plot_what='stimd', axis_handle=None)
    plots correlation between opsin expression and response magnitude for V1 and M2
    
    INPUTS:
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type    : str, 'dff' (default) or 'deconv'
        v1_data      : dict with V1 expression stats, output of get_opsin_expression_vs_response (optional, if not provided will call that method)
        m2_data      : dict with M2 expression stats, get_opsin_expression_vs_response (optional, if not provided will call that method)
        plot_what    : str, 'stimd' (default) or 'non_stimd'
        axis_handle  : axis handle for plotting (optional)
        
    OUTPUTS:
        expression_stats : dict with summary stats
        ax               : axis handle
    """
    
    # get data if necessary
    if v1_data is None:
        v1_data = get_opsin_expression_vs_response('V1', params=params, expt_type=expt_type, resp_type=resp_type)
    if m2_data is None:
        m2_data = get_opsin_expression_vs_response('M2', params=params, expt_type=expt_type, resp_type=resp_type)
    
    # stats
    expression_stats = dict()
    expression_stats['V1_summary'] = v1_data
    expression_stats['M2_summary'] = m2_data
    
    if plot_what != 'stimd' and plot_what != 'non_stimd':
        print('Unknown plot_what, returning analysis without plot')
        return expression_stats, None
    
    # plot
    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle
        
    # easy access variables
    x_v1  = v1_data['expression_levels']
    x_m2  = m2_data['expression_levels']
    y_v1  = v1_data['abs_resp_'+plot_what]
    cc_v1 = v1_data['cc_express_vs_'+plot_what+'_resp']
    p_v1  = v1_data['pval_express_vs_'+plot_what+'_resp']
    y_m2  = m2_data['abs_resp_'+plot_what]
    cc_m2 = m2_data['cc_express_vs_'+plot_what+'_resp']
    p_m2  = m2_data['pval_express_vs_'+plot_what+'_resp']
    x_v1  = x_v1[~np.isnan(y_v1)]
    y_v1  = y_v1[~np.isnan(y_v1)]
    x_m2  = x_m2[~np.isnan(y_m2)]
    y_m2  = y_m2[~np.isnan(y_m2)]
    
    # fit for display only
    olsfit_v1 = sm.OLS(y_v1,sm.add_constant(x_v1)).fit()
    x_hat_v1  = np.arange(np.min(np.concatenate((x_v1,x_m2))),np.max(np.concatenate((x_v1,x_m2)))+.1,.1)
    y_hat_v1  = olsfit_v1.predict(sm.add_constant(x_hat_v1))
    predci    = olsfit_v1.get_prediction(sm.add_constant(x_hat_v1)).summary_frame()
    ci_up_v1  = predci.loc[:,'mean_ci_upper']
    ci_low_v1 = predci.loc[:,'mean_ci_lower']
    
    olsfit_m2 = sm.OLS(y_m2,sm.add_constant(x_m2)).fit()
    x_hat_m2  = x_hat_v1
    y_hat_m2  = olsfit_m2.predict(sm.add_constant(x_hat_m2))
    predci    = olsfit_m2.get_prediction(sm.add_constant(x_hat_m2)).summary_frame()
    ci_up_m2  = predci.loc[:,'mean_ci_upper']
    ci_low_m2 = predci.loc[:,'mean_ci_lower']

    # finally plot everything
    ax.plot(x_v1,y_v1,'o',color=params['general_params']['V1_sh'],mew=0)
    ax.plot(x_hat_v1,y_hat_v1,'-',color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'])
    ax.plot(x_hat_v1,ci_up_v1,'--',color=params['general_params']['V1_cl'],lw=.4)
    ax.plot(x_hat_v1,ci_low_v1,'--',color=params['general_params']['V1_cl'],lw=.4)
    
    ax.plot(x_m2,y_m2,'o',color=params['general_params']['M2_sh'],mew=0)
    ax.plot(x_hat_m2,y_hat_m2,'-',color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'])
    ax.plot(x_hat_m2,ci_up_m2,'--',color=params['general_params']['M2_cl'],lw=.4)
    ax.plot(x_hat_m2,ci_low_m2,'--',color=params['general_params']['M2_cl'],lw=.4)

    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.text(xl[0]+.05,yl[1]*.9,'{}: r = {:1.2f}, p = {:1.2g}'.format(params['general_params']['V1_lbl'],cc_v1,p_v1),horizontalalignment='left',color=params['general_params']['V1_cl'])
    ax.text(xl[0]+.05,yl[1]*.82,'{}: r = {:1.2f}, p = {:1.2g}'.format(params['general_params']['M2_lbl'],cc_m2,p_m2),horizontalalignment='left',color=params['general_params']['M2_cl'])
    
    if 'zscore' in v1_data['expression_level_type']:
        ax.set_xlabel('Opsin expression level (z-score)')
    else:
        ax.set_xlabel('Opsin expression level (a.u.)')
    if plot_what == 'stimd':
        ax.set_ylabel("Max abs(response), stim'd cells (z-score)")
    else:
        ax.set_ylabel("Max abs(response), non-stim'd cells (z-score)")

    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return expression_stats, ax     

# ---------------
# %% get single-trial opto-triggered responses for an area and experiment type
def get_single_trial_data(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', eg_ids=None, signif_only=True, which_neurons='non_stimd', relax_timing_criteria=params['xval_relax_timing_criteria']):
    
    """
    get_single_trial_data(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', eg_ids=None, signif_only=True, which_neurons='non_stimd', relax_timing_criteria=params['xval_relax_timing_criteria'])
    retrieves single-trial opto-triggered responses for a given area and experiment type
    
    INPUTS:
        area                  : str, 'V1' or 'M2' (default is 'M2')
        params                : dict, analysis parameters (default is params from top of this script)
        expt_type             : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type             : str, 'dff' (default) or 'deconv'
        eg_ids                : list of int, experiment group ids to restrict to (optional, default is None)
        signif_only           : bool, if True only include significant neurons (default is True)
        which_neurons         : str, 'non_stimd' (default), 'all', 'stimd'
        relax_timing_criteria : bool, if True relax timing criteria (default is in params['xval_relax_timing_criteria']). 
                                This will fetch from a different param set that doesn't require the timing criteria
        
    OUTPUTS:
        trial_data : dict with summary data and single-trial responses
    """
    
    start_time      = time.time()
    print('Fetching opto-triggered trials...')
    
    # get relevant keys
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    
    # restrict to only desired rec/stim if applicable
    if eg_ids is not None:
        if isinstance(eg_ids,list) == False:
            eg_ids = [eg_ids]
        expt_keys = list(np.array(expt_keys)[np.array(eg_ids).astype(int)])
    
    # loop through keys to fetch the responses
    trigdff_param_set_id = params['trigdff_param_set_id_{}'.format(resp_type)]
    if relax_timing_criteria:
        trigdff_inclusion_param_set_id = params['trigdff_inclusion_param_set_id_notiming']
    else:
        trigdff_inclusion_param_set_id = params['trigdff_inclusion_param_set_id']
    
    trial_resps = list()
    trial_ids   = list()
    stim_ids    = list()
    roi_ids     = list()
    peak_ts     = list()
    coms        = list()
    t_axes      = list()
    num_expt    = len(expt_keys)
    for ct, ikey in enumerate(expt_keys):
        print('     {} of {}...'.format(ct+1,num_expt))
        this_key = {'subject_fullname' : ikey['subject_fullname'], 
                    'session_date': ikey['session_date'], 
                    'trigdff_param_set_id': trigdff_param_set_id, 
                    'trigdff_inclusion_param_set_id': trigdff_inclusion_param_set_id
                    }
        
        # do selecion at the fetching level for speed
        if which_neurons == 'stimd':
            # stimd neurons bypass inclusion criteria
            avg_keys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('KEY')
            tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (twop_opto_analysis.TrigDffTrial & avg_keys).fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id')
        elif which_neurons == 'non_stimd':
            avg_keys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=0').fetch('KEY')
            sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & avg_keys).fetch('is_significant', 'is_included', 'KEY')
            idx  = np.argwhere(np.array(incl)==1).flatten()
            keys = list(np.array(keys)[idx])
            sig  = list(np.array(sig)[idx])
            
            if signif_only:
                idx  = np.argwhere(np.array(sig)==1).flatten()
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx]) 
            
            tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (twop_opto_analysis.TrigDffTrial & keys).fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id')
        else:
            print('code not implemented for this category of which_neurons, returning nothing')
            return None

        # pick trough or peak time, whichever is higher magnitude
        maxmin_t = list()
        for iNeuron in range(len(trought)):
            if maxmin[iNeuron] < 0:
                maxmin_t.append(trought[iNeuron])
            else:
                maxmin_t.append(peakt[iNeuron])
                
        # flatten lists
        [trial_resps.append(trial) for trial in trials]
        [trial_ids.append(int(tid)) for tid in tids]
        [stim_ids.append(int(sid)) for sid in sids]
        [roi_ids.append(int(rid+(ct*10000))) for rid in rids] # 10000 is arbitrary experiment increment to make roi_ids unique
        [t_axes.append(t) for t in ts]
        [coms.append(co) for co in com]
        [peak_ts.append(pt) for pt in maxmin_t]
            
    # convert to arrays for easy indexing, trial and time vectors remain lists
    trial_ids = np.array(trial_ids)
    stim_ids  = np.array(stim_ids)
    roi_ids   = np.array(roi_ids)
    coms      = np.array(coms)
    peak_ts   = np.array(peak_ts)
        
    # collect summary data   
    trial_data = {
                'trig_dff_trials'         : trial_resps, 
                'trial_ids'               : trial_ids, 
                'time_axis_sec'           : t_axes,
                'signif_only'             : signif_only,
                'stim_ids'                : stim_ids, 
                'roi_ids'                 : roi_ids, 
                'com_sec'                 : coms, 
                'peak_or_trough_time_sec' : peak_ts, 
                'relax_timing_criteria'   : relax_timing_criteria,
                'which_neurons'           : which_neurons,
                'response_type'           : resp_type, 
                'experiment_type'         : expt_type, 
                'analysis_params'         : deepcopy(params)
                }
    
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    return trial_data

# ---------------
# %% cross-validate response timing
def xval_trial_data(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', signif_only=True, which_neurons='non_stimd', trial_data=None, rng=None):

    """
    xval_trial_data(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', signif_only=True, which_neurons='non_stimd', trial_data=None, rng=None)
    cross-validates response timing for a given area and experiment type
    
    INPUTS:
        area         : str, 'V1' or 'M2' (default is 'M2')
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard', 'short_stim', 'high_trial_count' (default), 'multi_cell'
        resp_type    : str, 'dff' (default) or 'deconv'
        signif_only  : bool, if True only include significant neurons (default is True)
        which_neurons: str, 'non_stimd' (default), 'all', 'stimd'
        trial_data   : dict with trial data, output of get_single_trial_data (optional, if not provided will call that method)
        rng          : numpy random number generator (optional)
        
    OUTPUTS:
        timing_stats : dict with summary stats and analysis results
    """
    # set random seed and delete low /  high tau values if applicable
    start_time      = time.time()
    print('Cross-validating reponse timing...')
    if rng is None:
        rng = np.random.default_rng(seed=params['random_seed'])
        
    # get data if necessary
    if trial_data is None:    
        trial_data = get_single_trial_data(area=area, params=params, expt_type=expt_type, resp_type=resp_type, signif_only=signif_only, which_neurons=which_neurons, relax_timing_criteria=params['xval_relax_timing_criteria'])
        
    # loop through rois and stims    
    sem_overall  = list()
    median_half1 = list()
    median_half2 = list()
    unique_rois  = list(np.unique(trial_data['roi_ids']))
    for roi in unique_rois:
        ridx         = trial_data['roi_ids']==roi
        these_trials = trial_data['trial_ids'][ridx]
        these_stims  = trial_data['stim_ids'][ridx]
        coms         = trial_data['com_sec'][ridx]
        peaks        = trial_data['peak_or_trough_time_sec'][ridx]
        trial_dffs   = list(np.array(trial_data['trig_dff_trials'])[ridx])
        t_axes       = list(np.array(trial_data['time_axis_sec'])[ridx])
        unique_stims = list(np.unique(these_stims))
        
        # take random halves of trials and compare timing stats for each 
        for stim in unique_stims:
            tidx        = these_trials[these_stims==stim]-1
            tidx_shuff  = deepcopy(tidx)
            ntrials     = np.size(tidx)
            timing_set1 = np.zeros(params['xval_num_iter'])
            timing_set2 = np.zeros(params['xval_num_iter'])
            taxis       = t_axes[tidx[0]]
            frame_int   = np.diff(taxis)[0]
            
            # randomly permute trials 
            for iShuff in range(params['xval_num_iter']):
                rng.shuffle(tidx_shuff)
                half1 = tidx_shuff[:np.floor(ntrials/2).astype(int)]
                half2 = tidx_shuff[np.floor(ntrials/2).astype(int)+1:]
                
                # get averages of peak (com), or compute avgs for each trial split and peak from there
                if params['xval_recompute_timing']:
                    # trial avgs
                    trial_avg1 = np.zeros(np.size(t_axes[tidx[0]]))
                    for idx in list(half1):
                        trial_avg1 += trial_dffs[idx]
                    trial_avg1 = trial_avg1 / len(half1)
                    
                    trial_avg2 = np.zeros(np.size(t_axes[tidx[0]]))
                    for idx in list(half2):
                        trial_avg2 += trial_dffs[idx]
                    trial_avg2 = trial_avg2 / len(half2)
                    
                    # smooth for peak (com), extract that
                    smoothed1 = general_stats.moving_average(trial_avg1,num_points=np.round(0.2/frame_int).astype(int)).flatten()
                    if np.sum(np.isnan(smoothed1)) > 0:
                        post_idx = int(np.argwhere(np.isnan(smoothed1))[-1]+1)
                    else:
                        post_idx = int(np.argwhere(taxis>0)[0])
                    com1, peak1, trough1 = twop_opto_analysis.response_time_stats(smoothed1[post_idx:],taxis[post_idx:])
                    
                    smoothed2 = general_stats.moving_average(trial_avg2,num_points=np.round(0.2/frame_int).astype(int)).flatten()
                    com2, peak2, trough2 = twop_opto_analysis.response_time_stats(smoothed2[post_idx:],taxis[post_idx:])
                    
                    # collect relevant stat
                    if params['xval_timing_metric'] == 'peak':
                        # take peak or trough, whichever is larger
                        if np.max(smoothed1+smoothed2) > np.abs(np.min(smoothed1+smoothed2)):
                            timing_set1[iShuff] = peak1
                            timing_set2[iShuff] = peak2
                            if iShuff == 0:
                                sem_overall.append(np.std([peak1,peak2])/np.sqrt(len(tidx)-1))
                        else:
                            timing_set1[iShuff] = trough1
                            timing_set2[iShuff] = trough2
                            if iShuff == 0:
                                sem_overall.append(np.std([trough1,trough2])/np.sqrt(len(tidx)-1))
                        
                    elif params['xval_timing_metric'] == 'com':
                        timing_set1[iShuff] = com1
                        timing_set2[iShuff] = com2
                        if iShuff == 0:
                            sem_overall.append(np.std([com1,com2])/np.sqrt(len(tidx)-1))
                        
                    else:
                        print('unknown parameter value for timing metric, returning nothing')
                        return None
                
                # or just take a median of pre-computed trial-by-trial features    
                else:
                    if params['xval_timing_metric'] == 'peak':
                        timing_set1[iShuff] = np.nanmedian(peaks[half1])
                        timing_set2[iShuff] = np.nanmedian(peaks[half2])
                        if iShuff == 0:
                            sem_overall.append(np.std(peaks)/np.sqrt(len(tidx)-1))
                        
                    elif params['xval_timing_metric'] == 'com':
                        timing_set1[iShuff] = np.nanmedian(coms[half1])
                        timing_set2[iShuff] = np.nanmedian(coms[half2])
                        if iShuff == 0:
                            sem_overall.append(np.std(coms)/np.sqrt(len(tidx)-1))
                        
                    else:
                        print('unknown parameter value for timing metric, returning nothing')
                        return None
            
            # collect median metric for each half
            median_half1.append(np.nanmedian(timing_set1))
            median_half2.append(np.nanmedian(timing_set2))
            
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    xval_results = {
                'timing_metric'     : params['xval_timing_metric'], 
                'trial_sem'         : np.array(sem_overall).flatten(), 
                'median_trialset1'  : np.array(median_half1).flatten(),
                'median_trialset2'  : np.array(median_half2).flatten(), 
                'response_type'     : resp_type, 
                'experiment_type'   : expt_type, 
                'which_neurons'     : which_neurons,
                'analysis_params'   : deepcopy(params)
                }
    
    
    return xval_results, trial_data

# ---------------
# %% plot response timing cross-validation results
def plot_trial_xval(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', signif_only=True, which_neurons='non_stimd', xval_results=None, rng=None, axis_handle=None, fig_handle=None, trial_data=None):

    """
    plot_trial_xval(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', signif_only=True, which_neurons='non_stimd', xval_results=None, rng=None, axis_handle=None, fig_handle=None, trial_data=None)
    plots response timing cross-validation results
    
    INPUTS:
        area         : str, 'V1' or 'M2' (default)
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard', 'short_stim', 'high_trial_count' (default), 'multi_cell'
        resp_type    : str, 'dff' (default) or 'deconv'
        signif_only  : bool, if True only include significant neurons (default is True)
        which_neurons: str, 'non_stimd' (default), 'all', 'stimd'
        xval_results : dict with xval results, output of xval_trial_data (optional, if not provided will call that method)
        rng          : numpy random number generator (optional)
        axis_handle  : axis handle for plotting (optional)
        fig_handle   : figure handle for plotting (optional)
        trial_data   : dict with trial data, output of get_single_trial_data (optional, if not provided will call that method)
        
    OUTPUTS:
        ax           : axis handle
        fig          : figure handle
        xval_results : dict with xval results
    """
    
    # run analysis if necessary
    if xval_results is None:
        xval_results, _ = xval_trial_data(area=area, params=params, expt_type=expt_type, resp_type=resp_type, signif_only=signif_only, which_neurons=which_neurons, rng=rng, trial_data=trial_data)
        
    # plot
    if axis_handle is None:
        fig = plt.figure()
        ax  = plt.gca()
    else:
        ax  = axis_handle
        fig = fig_handle
        
    half1  = xval_results['median_trialset1']  
    half2  = xval_results['median_trialset2']    
    sem    = xval_results['trial_sem']
    
    xy_lim       = [np.min(np.concatenate((half1,half2)))-.5, np.max(np.concatenate((half1,half2)))+.5]  
    this_cmap    = plt.colormaps.get_cmap('copper')
    this_cmap.set_bad(color='w')
    
    ax.plot(xy_lim,xy_lim,'--',color=[.8,.8,.8])
    ax.scatter(x=half1,y=half2,c=sem,cmap=this_cmap,edgecolors=[.5,.5,.5],linewidths=.5)
    ax.set_xlim(xy_lim)
    
    if 'peak' in xval_results['timing_metric']:
        ax.set_xlabel('Median peak time, half 1 (sec)')
        ax.set_ylabel('Median peak time, half 2 (sec)')
    else:
        ax.set_xlabel('Median COM time, half 1 (sec)')
        ax.set_ylabel('Median COM time, half 2 (sec)')
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    fig.colorbar(ax.scatter(x=half1,y=half2,c=sem,cmap=this_cmap),label='S.E.M. (sec)')
        
    return ax, fig, xval_results
        
# ---------------
# %% plot average response heatmap
def plot_avg_response_heatmap(area, params=params, expt_type='standard', resp_type='dff', signif_only=True, which_neurons='non_stimd', avg_data=None, axis_handle=None, fig_handle=None, norm_type='minmax'):

    """
    plot_avg_response_heatmap(area, params=params, expt_type='standard', resp_type='dff', signif_only=True, which_neurons='non_stimd', avg_data=None, axis_handle=None, fig_handle=None, norm_type='minmax')
    Plots average response heatmap for an area for each roi, sorted by peak time.
    
    INPUTS:
    area: str, 'V1' or 'M2'
    params: dict, parameters dictionary
    expt_type: str, 'standard', 'high_trial_count', 'short_stim', 'multi_cell'
    """
    if avg_data is None:
        avg_data = get_avg_trig_responses(area, params=params, expt_type=expt_type, resp_type=resp_type, signif_only=signif_only, which_neurons=which_neurons)
        
    # sort by peak time
    idx         = np.argsort(avg_data['peak_times_sec']).flatten()
    resp_mat    = deepcopy(avg_data['trig_dff_avgs'])[idx,:]
    idx         = np.argsort(avg_data['peak_times_sec'])
    num_neurons = np.size(resp_mat,axis=0)
    
    # normalize
    if norm_type == 'minmax':
        for iNeuron in range(num_neurons):
            resp_mat[iNeuron,:] = resp_mat[iNeuron,:] - np.nanmin(resp_mat[iNeuron,:]) 
            resp_mat[iNeuron,:] = resp_mat[iNeuron,:] / np.nanmax(resp_mat[iNeuron,:])
        cm_name = 'bone'
        lbl     = 'Normalized response'
    elif norm_type == 'absmax':
        for iNeuron in range(num_neurons):
            resp_mat[iNeuron,:] = resp_mat[iNeuron,:] / np.nanmax(np.abs(resp_mat[iNeuron,:]))
        cm_name = 'coolwarm'
        lbl     = 'Normalized response'
    elif norm_type == 'none':
        cm_name = 'bone'
        lbl     = 'Response (z-score)'
    else:
        print('unknown normalization type, returning nothing')
        return None, None, None
    
    # time axis
    t_axis = avg_data['time_axis_sec']
    xticks = range(-2,np.ceil(t_axis[-1]).astype(int),2)
    xpos = list()
    for x in xticks:
        xpos.append(np.argwhere(t_axis>=x).flatten()[0])
        
    # plot
    if axis_handle is None:
        fig = plt.figure()
        ax  = plt.gca()
    else:
        ax  = axis_handle
        fig = fig_handle

    fig.colorbar(ax.imshow(resp_mat,cmap=cm_name,aspect='auto'),label=lbl)
    ax.set_xticks(np.array(xpos).astype(int))
    ax.set_xticklabels(xticks)
    ax.set_xlabel('Time from stim (sec)')
    ax.set_ylabel('Significant responses (neurons*stim)')
    
    ax.set_title(params['general_params']['{}_lbl'.format(area)])
    
    return ax, fig, avg_data

# ---------------
# %% plot grand average of response time course
def plot_response_grand_average(params=params, expt_type='standard', resp_type='dff', signif_only=True, which_neurons='non_stimd', v1_data=None, m2_data=None, axis_handle=None, norm_type='peak'):

    """
    plot_response_grand_average(area, params=params, expt_type='standard', resp_type='dff', signif_only=True, which_neurons='non_stimd', avg_data=None, axis_handle=None, fig_handle=None)
    Plots grand average response time course for both areas on the same axis
    
    INPUTS:
        params        : dict, parameters dictionary
        expt_type     : str, 'standard' (default), 'high_trial_count', 'short_stim', 'multi_cell'
        resp_type     : str, 'dff' (default) or 'deconv'
        signif_only   : bool, if True only include significant neurons (default is True)
        which_neurons : str, 'non_stimd' (default), 'all', 'stimd'
        v1_data       : dict with average data, output of get_avg_trig_responses (optional, if not provided will call that method)
        m2_data       : dict with average data, output of get_avg_trig_responses (optional, if not provided will call that method)
        axis_handle   : axis handle for plotting (optional)
        norm_type     : str, 'peak' (default), 't0', 'minmax', 'absmax' , 'none'
        
    OUTPUTS:
        ax            : axis handle
        v1_data       : dict with average data
        m2_data       : dict with average data
    """
    
    # get data if necessary (takes a while)
    if v1_data is None:
        v1_data = get_avg_trig_responses('V1', params=params, expt_type=expt_type, resp_type=resp_type, signif_only=signif_only, which_neurons=which_neurons)
    if m2_data is None:
        m2_data = get_avg_trig_responses('M2', params=params, expt_type=expt_type, resp_type=resp_type, signif_only=signif_only, which_neurons=which_neurons) 
    
    # normalize
    resp_mat_v1 = deepcopy(v1_data['trig_dff_avgs'])
    resp_mat_m2 = deepcopy(m2_data['trig_dff_avgs'])
    num_v1      = np.size(resp_mat_v1,axis=0)
    num_m2      = np.size(resp_mat_m2,axis=0)
    t_axis_v1   = v1_data['time_axis_sec']
    t_axis_m2   = m2_data['time_axis_sec']
    
    if norm_type == 'minmax': # between 0 and 1
        for iNeuron in range(num_v1):
            resp_mat_v1[iNeuron,:] = resp_mat_v1[iNeuron,:] - np.nanmin(resp_mat_v1[iNeuron,:]) 
            resp_mat_v1[iNeuron,:] = resp_mat_v1[iNeuron,:] / np.nanmax(resp_mat_v1[iNeuron,:])
        for iNeuron in range(num_m2):
            resp_mat_m2[iNeuron,:] = resp_mat_m2[iNeuron,:] - np.nanmin(resp_mat_m2[iNeuron,:]) 
            resp_mat_m2[iNeuron,:] = resp_mat_m2[iNeuron,:] / np.nanmax(resp_mat_m2[iNeuron,:])
        lbl     = 'Average normalized response'
        
    elif norm_type == 'absmax': # to max abs response
        for iNeuron in range(num_v1):
            resp_mat_v1[iNeuron,:] = resp_mat_v1[iNeuron,:] / np.nanmax(np.abs(resp_mat_v1[iNeuron,:]))
        for iNeuron in range(num_m2):
            resp_mat_m2[iNeuron,:] = resp_mat_m2[iNeuron,:] / np.nanmax(np.abs(resp_mat_m2[iNeuron,:]))
        lbl     = 'Average normalized response'
        
    elif norm_type == 'peak': # max response
        for iNeuron in range(num_v1):
            resp_mat_v1[iNeuron,:] = resp_mat_v1[iNeuron,:] / np.nanmax(resp_mat_v1[iNeuron,:])
        for iNeuron in range(num_m2):
            resp_mat_m2[iNeuron,:] = resp_mat_m2[iNeuron,:] / np.nanmax(resp_mat_m2[iNeuron,:])
        lbl     = 'Average normalized response'
        
    elif norm_type == 't0': # to first datapoint after shutter opens
        t0idx_v1 = np.argwhere(np.isnan(np.sum(resp_mat_v1,axis=0))).flatten()[-1]+1
        t0idx_m2 = np.argwhere(np.isnan(np.sum(resp_mat_m2,axis=0))).flatten()[-1]+1
        for iNeuron in range(num_v1):
            resp_mat_v1[iNeuron,:] = resp_mat_v1[iNeuron,:] / resp_mat_v1[iNeuron,t0idx_v1]
        for iNeuron in range(num_m2):
            resp_mat_m2[iNeuron,:] = resp_mat_m2[iNeuron,:] / resp_mat_m2[iNeuron,t0idx_m2]
        lbl     = 'Average normalized response'
        
    elif norm_type == 'none':
        lbl     = 'Average response (z-score)'
        
    else:
        print('unknown normalization type, returning nothing')
        return None, None, None
        
    # grand average
    v1_avg = np.nanmean(resp_mat_v1 / np.nanmax(resp_mat_v1),axis=0)
    v1_sem = np.nanstd(resp_mat_v1 / np.nanmax(resp_mat_v1),axis=0)/np.sqrt(num_v1-1)
    m2_avg = np.nanmean(resp_mat_m2 / np.nanmax(resp_mat_m2),axis=0)
    m2_sem = np.nanstd(resp_mat_m2/ np.nanmax(resp_mat_m2),axis=0)/np.sqrt(num_m2-1)
        
    # plot
    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle

    ax.fill_between(t_axis_v1,v1_avg-v1_sem,v1_avg+v1_sem,color=params['general_params']['V1_sh'])
    ax.plot(t_axis_v1,v1_avg,'-',color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'])
    ax.fill_between(t_axis_m2,m2_avg-m2_sem,m2_avg+m2_sem,color=params['general_params']['M2_sh']) 
    ax.plot(t_axis_m2,m2_avg,'-',color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'])
    
    ax.legend()
    ax.set_xlabel('Time from stim (sec)')
    ax.set_ylabel(lbl)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return ax, v1_data, m2_data

# ---------------
# %% analyze opto response probability vs tau
def opto_vs_tau(area, params=params, expt_type='standard', resp_type='dff', dff_type='residuals_dff', opto_data=None, tau_data=None):
    
    """
    opto_vs_tau(area, params=params, expt_type='standard', resp_type='dff', dff_type = 'residuals_dff', opto_data=None, tau_data=None)
    plots response timing cross-validation results
    
    INPUTS:
        area         : str, 'V1' or 'M2' 
        params       : dict, analysis parameters (default is params from top of this script)
        expt_type    : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type    : str, 'dff' (default) or 'deconv'
        dff_type     : str, 'residuals_dff' (default), 'noGlm_dff', 'residuals_deconv'. Type use to calculate tau
        opto_data    : dict with response stats, output of get_full_resp_stats (optional, if not provided will call that method)
        tau_data     : dict with tau stats, output of analyzeSpont2P.get_tau_from_roi_keys (optional, if not provided will call that method)
                       note that rois of opto_data and tau_data need to match exactly, enforced within this method
        prob_by_expt : bool, if True computes response probability by experiment and then averages (default is False)
        
    OUTPUT:
        analysis_results : dict with results of tau vs opto analysis
    """
    
    # get data for opto. this must contain both stimd and non-stimd cells
    if opto_data is None:
        opto_data = get_full_resp_stats(area=area, params=params, expt_type=expt_type, resp_type=resp_type, which_neurons='all')
    
    # here we need to be clunky and loop to enforce correspondence between opto and tau data
    if tau_data is None:
        start_time = time.time()
        print('Fetching tau data...')
        
        sess_ids    = analyzeSpont2P.sess_ids_from_tau_keys(opto_data['roi_keys']) 
        stim_ids    = deepcopy(opto_data['stim_ids'])
        taus = list()
        is_good_tau = list()

        for sess in np.unique(sess_ids):
            this_sess = sess_ids==sess
            unique_stim = np.unique(stim_ids[this_sess])
            for stim in unique_stim:
                this_stim = np.logical_and(this_sess,stim_ids==stim)
                keys = list(np.array(opto_data['roi_keys'])[this_stim])
                td  = analyzeSpont2P.get_tau_from_roi_keys(keys, params=params, dff_type=dff_type, verbose=False)
                [taus.append(t) for t in td['taus']]
                [is_good_tau.append(ig) for ig in td['is_good_tau']]
                
        tau_data = {'taus':np.array(taus).flatten(),'is_good_tau':np.array(is_good_tau).flatten()}
        
        end_time = time.time()
        print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    else:
        sess_ids    = analyzeSpont2P.sess_ids_from_tau_keys(opto_data['roi_keys']) 
        stim_ids    = deepcopy(opto_data['stim_ids'])
        
    # easy access variables    
    is_stimd    = deepcopy(opto_data['is_stimd'])
    is_sig      = deepcopy(opto_data['is_sig'])
    peak_ts     = deepcopy(opto_data['max_or_min_times_sec'])
    peak_mag    = deepcopy(opto_data['max_or_min_vals'])
    tau         = deepcopy(tau_data['taus'])
    is_good_tau = deepcopy(tau_data['is_good_tau'])
    
    # select out very long taus if desired
    if params['tau_vs_opto_max_tau'] is not None:
        is_good_tau[tau>params['tau_vs_opto_max_tau']] = 0
        
    # do median split on tau
    tau_th      = np.median(tau[is_good_tau==1])
    is_short    = tau < tau_th
    
    # response properties by tau (need to implement peak width)
    bins     = params['tau_bins']
    num_bins = np.size(bins)-1
    peakt_by_tau_avg  = np.zeros(num_bins)
    peakt_by_tau_sem  = np.zeros(num_bins)
    peakt_by_tau_expt = [None]*num_bins
    peakm_by_tau_avg  = np.zeros(num_bins)
    peakm_by_tau_sem  = np.zeros(num_bins)
    peakm_by_tau_expt = [None]*num_bins
    for iBin in range(num_bins):
        # idx     = np.logical_and(is_good_tau==1,np.logical_and(tau>bins[iBin], tau<=bins[iBin+1]))
        idx     = np.logical_and(tau>bins[iBin], tau<=bins[iBin+1])

        idx     = np.logical_and(is_stimd==0,np.logical_and(is_sig==1,idx))
        sem_den = np.sqrt(np.sum(idx==1)-1)
        peakt_by_tau_avg[iBin] = np.mean(peak_ts[idx])
        peakt_by_tau_sem[iBin] = (np.std(peak_ts[idx],ddof=1))/sem_den
        peakm_by_tau_avg[iBin] = np.mean(peak_mag[idx])
        peakm_by_tau_sem[iBin] = (np.std(peak_mag[idx],ddof=1))/sem_den
        sess = np.unique(sess_ids[idx])
        peaks = list()
        mags  = list()
        for s in sess:
            idx_sess = np.logical_and(sess_ids==s,idx)
            peaks.append(peak_ts[idx_sess])
            mags.append(peak_mag[idx_sess])
        peakt_by_tau_expt[iBin] = peaks
        peakm_by_tau_expt[iBin] = mags
    
    # get responding stats by stimd and non-stimd tau, overall and across time
    unique_sess = list(np.unique(sess_ids))
    ct_short    = 0
    ct_long     = 0
    tau_mat_counts       = np.zeros((2,2))
    tau_mat              = np.zeros((2,2))
    tau_t_bins           = params['tau_by_time_bins']
    tau_mat_t_counts     = [np.zeros((2,2))]*(len(tau_t_bins)-1)
    tau_mat_t            = [np.zeros((2,2))]*(len(tau_t_bins)-1) # normed by cells in each bin
    tau_mat_t_by_overall = [np.zeros((2,2))]*(len(tau_t_bins)-1) # normed by overall responding cells
    t_counts             = [np.zeros(2)]*(len(tau_t_bins)-1)

    if params['tau_vs_opto_do_prob_by_expt']: # take averages across expts
        for sess in unique_sess:
            unique_stim = list(np.unique(stim_ids[sess_ids==sess]))
            for stim in unique_stim:
                these_cells   = np.logical_and(np.logical_and(sess_ids==sess, stim_ids==stim),is_good_tau==1)
                stimd_idx     = np.logical_and(is_stimd==1, these_cells)
                non_stimd_idx = np.logical_and(np.logical_and(is_stimd==0, these_cells),is_sig==1)
                
            if np.sum(stimd_idx) == 0:
                continue
            
            # divide by tau median and compute response prob
            total_good_cells = np.sum(these_cells==1)-np.sum(stimd_idx)
            if is_short[stimd_idx]:
                ct_short += 1 # this is just incrementing experiments for averaging
                mat_row   = 0
            else:
                ct_long += 1
                mat_row  = 1
            
            # overall resp prob by tau
            short_idx = np.logical_and(is_short==1,non_stimd_idx==1)
            long_idx  = np.logical_and(is_short==0,non_stimd_idx==1)
            tau_mat_counts[mat_row,0] += np.sum(short_idx) 
            tau_mat_counts[mat_row,1] += np.sum(long_idx) 
            tau_mat[mat_row,0] += np.sum(short_idx) / total_good_cells
            tau_mat[mat_row,1] += np.sum(long_idx) / total_good_cells
            
            # now do it by time by restricting by peak time
            for iBin in range(len(tau_mat_t_counts)):
                peak_idx    = np.logical_and(peak_ts>tau_t_bins[iBin], peak_ts<=tau_t_bins[iBin+1])
                short_idx_t = np.logical_and(peak_idx,short_idx)
                long_idx_t  = np.logical_and(peak_idx,long_idx)
                total_t     = np.sum(np.logical_and(peak_idx,these_cells))-np.sum(stimd_idx)
                tau_mat_t[iBin][mat_row,0] += np.sum(short_idx_t==1) / total_t
                tau_mat_t[iBin][mat_row,1] += np.sum(long_idx_t==1) / total_t
                tau_mat_t_by_overall[iBin][mat_row,0] += np.sum(short_idx_t==1) / total_good_cells
                tau_mat_t_by_overall[iBin][mat_row,1] += np.sum(long_idx_t==1) / total_good_cells
                tau_mat_t_counts[iBin][mat_row,0] += np.sum(short_idx_t==1) 
                tau_mat_t_counts[iBin][mat_row,1] += np.sum(long_idx_t==1)

        # multiply by counts to get average numbers (from average probs)
        tau_mat[0,:] = tau_mat[0,:] / ct_short
        tau_mat[1,:] = tau_mat[1,:] / ct_long
        tau_mat_counts[0,:] = tau_mat_counts[0,:] / ct_short
        tau_mat_counts[1,:] = tau_mat_counts[1,:] / ct_long
        for iBin in range(len(tau_mat_t)):
            tau_mat_t[iBin][0,:] = tau_mat_t[iBin][0,:] / ct_short
            tau_mat_t[iBin][1,:] = tau_mat_t[iBin][1,:] / ct_long
            tau_mat_t_counts[iBin][0,:] = tau_mat_t_counts[iBin][0,:] / ct_short
            tau_mat_t_counts[iBin][1,:] = tau_mat_t_counts[iBin][1,:] / ct_long
            tau_mat_t_by_overall[iBin][0,:] = tau_mat_t_by_overall[iBin][0,:] / ct_short
            tau_mat_t_by_overall[iBin][1,:] = tau_mat_t_by_overall[iBin][1,:] / ct_long
            
    else: # just do overall probabilities grouped across experiments
        # vector of taus for stimd cells corersponding to cells in flattened vector
        tau_stimd   = np.zeros(np.size(tau))-1
        for sess in unique_sess:
            unique_stim = list(np.unique(stim_ids[sess_ids==sess]))
            for stim in unique_stim:
                these_cells = np.logical_and(sess_ids==sess, stim_ids==stim)
                stimd_idx   = np.logical_and(is_stimd==1, these_cells)
                tau_stimd[these_cells] = tau[stimd_idx]

        # overall resp prob by tau
        long_tau_stimd     = np.logical_and(tau_stimd > tau_th,is_good_tau==1)
        short_tau_stimd    = np.logical_and(tau_stimd <= tau_th,is_good_tau==1)
        long_tau_nonstimd  = np.logical_and(np.logical_and(is_sig==1,np.logical_and(tau > tau_th, is_stimd==0)),is_good_tau==1)
        short_tau_nonstimd = np.logical_and(np.logical_and(is_sig==1,np.logical_and(tau <= tau_th, is_stimd==0)),is_good_tau==1)
    
        tau_mat_counts[0,0] = np.sum(np.logical_and(short_tau_stimd,short_tau_nonstimd))
        tau_mat_counts[0,1] = np.sum(np.logical_and(short_tau_stimd,long_tau_nonstimd))
        tau_mat_counts[1,0] = np.sum(np.logical_and(long_tau_stimd,short_tau_nonstimd))
        tau_mat_counts[1,1] = np.sum(np.logical_and(long_tau_stimd,long_tau_nonstimd))
        
        # now do it by time by restricting by peak time
        for iBin in range(len(tau_mat_t_counts)):
            peak_idx    = np.logical_and(peak_ts>tau_t_bins[iBin], peak_ts<=tau_t_bins[iBin+1])
            short_tau_t = np.logical_and(peak_idx,short_tau_nonstimd)
            long_tau_t  = np.logical_and(peak_idx,long_tau_nonstimd)
            t_counts[iBin][0]           = np.sum(np.logical_and(peak_idx==1,short_tau_stimd==1))
            t_counts[iBin][1]           = np.sum(np.logical_and(peak_idx==1,long_tau_stimd==1))
            tau_mat_t_counts[iBin][0,0] = np.sum(np.logical_and(short_tau_stimd,short_tau_t))
            tau_mat_t_counts[iBin][0,1] = np.sum(np.logical_and(short_tau_stimd,long_tau_t))
            tau_mat_t_counts[iBin][1,0] = np.sum(np.logical_and(long_tau_stimd,short_tau_t))
            tau_mat_t_counts[iBin][1,1] = np.sum(np.logical_and(long_tau_stimd,long_tau_t))
            
        # count responding cells for normalization
        ct_long  = np.sum(long_tau_stimd==1)
        ct_short = np.sum(short_tau_stimd==1)
            
        # divide by counts to get average
        tau_mat[0,:] = tau_mat_counts[0,:] / ct_short
        tau_mat[1,:] = tau_mat_counts[1,:] / ct_long
        for iBin in range(len(tau_mat_t)):
            tau_mat_t[iBin][0,:] = tau_mat_t_counts[iBin][0,:] / t_counts[iBin][0]
            tau_mat_t[iBin][1,:] = tau_mat_t_counts[iBin][1,:] / t_counts[iBin][1]
            tau_mat_t_by_overall[iBin][0,:] = tau_mat_t_counts[iBin][0,:] / ct_short
            tau_mat_t_by_overall[iBin][1,:] = tau_mat_t_counts[iBin][1,:] / ct_long
    
    # collect analysis results
    analysis_results = {
                        'tau_xaxis_sec'               : bins[:-1]+np.diff(bins)[0]/2,
                        'peak_time_by_tau_mean'       : peakt_by_tau_avg,
                        'peak_time_by_tau_sem'        : peakt_by_tau_sem,
                        'peak_time_by_tau_expt'       : peakt_by_tau_expt,
                        'peak_mag_by_tau_mean'        : peakm_by_tau_avg,
                        'peak_mag_by_tau_sem'         : peakm_by_tau_sem,
                        'peak_mag_by_tau_expt'        : peakm_by_tau_expt,
                        'resp_prob_by_tau'            : tau_mat,
                        'resp_counts_by_tau'          : tau_mat_counts,
                        'tau_over_time_axis_sec'      : tau_t_bins[:-1]+np.diff(tau_t_bins)[0]/2,
                        'resp_prob_by_tau_over_time'  : tau_mat_t,
                        'resp_counts_by_tau_over_time': tau_mat_t_counts,
                        'resp_prob_by_tau_over_time_by_overall'  : tau_mat_t_by_overall,
                        'params'                      : deepcopy(params),
                        'tau_data'                    : tau_data,
                        'peaks'                       : peaks
                        }
            
    return analysis_results

# ---------------
# %% plot opto response properties vs tau
def plot_opto_vs_tau_comparison(area=None, plot_what='prob', params=params, expt_type='standard', resp_type='dff', dff_type='residuals_dff', tau_vs_opto_comp_summary=None, analysis_results_v1=None, analysis_results_m2=None, axis_handles=None):
    
    """
    plot_opto_vs_tau(area, params=params, expt_type='standard', resp_type='dff', dff_type = 'residuals_dff', opto_data=None, tau_data=None, axis_handle=None)
    plots response timing cross-validation results
    
    INPUTS:
        area       : str,  None (default), 'V1' or 'M2'. None will plot both areas 
        plot_what  : str, 'prob' (default), 'prob_by_time', 'prob_by_time_by_overall', 'peak_time', 'peak_mag'
        params     : dict, analysis parameters (default is params from top of this script)
        expt_type  : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type  : str, 'dff' (default) or 'deconv'
        dff_type   : str, 'residuals_dff' (default), 'noGlm_dff', 'residuals_deconv'. Type use to calculate tau
        tau_vs_opto_comp_summary : dict with results of tau vs opto analysis and area comparisons thereof (optional, if not provided will run analysis)
        analysis_results_v1      : dict with results of tau vs opto analysis (optional, if not provided will call that method)
        analysis_results_m2      : dict with results of tau vs opto analysis (optional, if not provided will call that method)
        axis_handles             : axis handle for plotting (optional). Certain plots require a list of multiple axis handles
        
    OUTPUT:
        ax               : axis handle(s)
        fig              : figure handle
        tau_vs_opto_comp_summary : dict with results of tau vs opto analysis and area comparisons thereof
    """
    
    # get data if necessary
    if tau_vs_opto_comp_summary is None:
        if area is None:
            if analysis_results_v1 is None:
                analysis_results_v1 = opto_vs_tau('V1', params=params, expt_type=expt_type, resp_type=resp_type, dff_type=dff_type)
            if analysis_results_m2 is None:
                analysis_results_m2 = opto_vs_tau('M2', params=params, expt_type=expt_type, resp_type=resp_type, dff_type=dff_type)
        elif area == 'V1':
            if analysis_results_v1 is None:
                analysis_results_v1 = opto_vs_tau('V1', params=params, expt_type=expt_type, resp_type=resp_type, dff_type=dff_type)
        elif area == 'M2':
            if analysis_results_m2 is None:
                analysis_results_m2 = opto_vs_tau('M2', params=params, expt_type=expt_type, resp_type=resp_type, dff_type=dff_type)
         
        # do area comparison
        tau_vs_opto_comp_summary = dict()
        tau_vs_opto_comp_summary['V1_results'] = analysis_results_v1
        tau_vs_opto_comp_summary['M2_results'] = analysis_results_m2
        if area is None:
            tau_vs_opto_comp_summary['stats'] = dict()

            # two-way ANOVAs for the tau-dependent metrics 
            # make it flattened lists for dataframe conversion first
            taus       = list()
            areas      = list()
            peak_ts    = list()
            peak_mags  = list()
            tau_vals   = list(analysis_results_v1['tau_xaxis_sec'])
            num_bins   = len(tau_vals)
            for iBin in range(num_bins):
                num_exp_v1 = len(analysis_results_v1['peak_time_by_tau_expt'][iBin])
                for iEx in range(num_exp_v1):
                    this_peak = list(analysis_results_v1['peak_time_by_tau_expt'][iBin][iEx])
                    this_mag  = list(analysis_results_v1['peak_mag_by_tau_expt'][iBin][iEx])
                    [peak_ts.append(ii) for ii in this_peak]
                    [peak_mags.append(ii) for ii in this_mag]
                    [taus.append(ii) for ii in [tau_vals[iBin]]*len(this_peak)]
                    [areas.append(ii) for ii in ['V1']*len(this_peak)]
            for iBin in range(num_bins):
                num_exp_m2 = len(analysis_results_m2['peak_time_by_tau_expt'][iBin])
                for iEx in range(num_exp_m2):
                    this_peak = list(analysis_results_m2['peak_time_by_tau_expt'][iBin][iEx])
                    this_mag  = list(analysis_results_m2['peak_mag_by_tau_expt'][iBin][iEx])
                    [peak_ts.append(ii) for ii in this_peak]
                    [peak_mags.append(ii) for ii in this_mag]
                    [taus.append(ii) for ii in [tau_vals[iBin]]*len(this_peak)]
                    [areas.append(ii) for ii in ['M2']*len(this_peak)]
                
            df  = pd.DataFrame({'area': areas, 
                                'taus': taus, 
                                'peak_ts'  : peak_ts, 
                                'peak_mag' : peak_mags
                                })

            tau_vs_opto_comp_summary['stats']['anova_peak_time_by_tau'] = pg.anova(data=df, dv='peak_ts', between=['area', 'taus'])
            tau_vs_opto_comp_summary['stats']['anova_peak_mag_by_tau']  = pg.anova(data=df, dv='peak_mag', between=['area', 'taus'])

            # chi2 tests for response probability
            # create a 3-way contingency table for chi-square tests
            chi2_data = {'area': ['V1']*4+['M2']*4,
                    'tau_stimd': ['short', 'short', 'long', 'long', 'short', 'short', 'long', 'long'],
                    'tau_nonstimd': ['short', 'long', 'short', 'long', 'short', 'long', 'short', 'long'],
                    'counts': list(analysis_results_v1['resp_counts_by_tau'].flatten())+list(analysis_results_m2['resp_counts_by_tau'].flatten())}
            chi2_df = pd.DataFrame(chi2_data)

            # chi-square test for area
            contingency_table = pd.crosstab(index=[chi2_df['tau_stimd'], chi2_df['tau_nonstimd']], columns=chi2_df['area'], values=chi2_df['counts'], aggfunc='sum')
            chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table)
            tau_vs_opto_comp_summary['stats']['chi2_area'] = {'chi2':chi2, 'p':p, 'dof':dof, 'expected':expected}

            # chi-square test for tau (responding)
            contingency_table = pd.crosstab(index=[chi2_df['area'], chi2_df['tau_stimd']], columns=chi2_df['tau_nonstimd'], values=chi2_df['counts'], aggfunc='sum')
            chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table)
            tau_vs_opto_comp_summary['stats']['chi2_tau_nonstimd'] = {'chi2':chi2, 'p':p, 'dof':dof, 'expected':expected}

            # chi-square test for tau (stimulated)
            contingency_table = pd.crosstab(index=[chi2_df['area'], chi2_df['tau_nonstimd']], columns=chi2_df['tau_stimd'], values=chi2_df['counts'], aggfunc='sum')
            chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table)
            tau_vs_opto_comp_summary['stats']['chi2_tau_stimd'] = {'chi2':chi2, 'p':p, 'dof':dof, 'expected':expected}
    
    else:
        if area is None:
            analysis_results_v1=tau_vs_opto_comp_summary['V1_results']
            analysis_results_m2=tau_vs_opto_comp_summary['M2_results']
        
        elif area == 'V1':
            analysis_results_v1=tau_vs_opto_comp_summary['V1_results']
        
        elif area == 'M2':
            analysis_results_m2=tau_vs_opto_comp_summary['M2_results']
            
    # plot (either two areas or one, lots of contingencies)
    # plot overall response probability by tau
    if plot_what == 'prob': 
        # multi-area case
        if area is None:
            # handle axes
            if axis_handles is None:
                fig = plt.figure()
                ax  = [plt.subplot(121), plt.subplot(122)]
            else:
                if len(axis_handles) != 2:
                    print('need two axis handles for this plot, returning just the stats')
                    return None, None, tau_vs_opto_comp_summary
                ax  = axis_handles
                fig = axis_handles[0].get_figure()



            if params['prob_plot_same_scale']:
                vmax = np.max([np.max(analysis_results_v1['resp_prob_by_tau']),np.max(analysis_results_m2['resp_prob_by_tau'])])
                vmin = np.min([np.min(analysis_results_v1['resp_prob_by_tau']),np.min(analysis_results_m2['resp_prob_by_tau'])])
                plt.colorbar(ax[0].imshow(analysis_results_v1['resp_prob_by_tau'],aspect='auto',cmap='bone',vmax=vmax,vmin=vmin),label='Response probability')
                plt.colorbar(ax[1].imshow(analysis_results_m2['resp_prob_by_tau'],aspect='auto',cmap='bone',vmax=vmax,vmin=vmin),label='Response probability')
            else:
                plt.colorbar(ax[0].imshow(analysis_results_v1['resp_prob_by_tau'],aspect='auto',cmap='bone',label='Response probability'),label='Response probability')
                plt.colorbar(ax[1].imshow(analysis_results_m2['resp_prob_by_tau'],aspect='auto',cmap='bone',label='Response probability'),label='Response probability')
            
            ax[0].set_title(params['general_params']['V1_lbl'])
            ax[1].set_title(params['general_params']['M2_lbl'])
            for iPlot in range(2):
                ax[iPlot].set_xticks([0,1])
                ax[iPlot].set_yticks([0,1])
                ax[iPlot].set_xticklabels(['short $\\tau$','long $\\tau$'])
                ax[iPlot].set_yticklabels(['short $\\tau$','long $\\tau$'])
                ax[iPlot].set_xlabel('Responding neurons')
                ax[iPlot].set_ylabel("Stim'd neurons")
                
            ax[0].text(1.5,-.5,'p(area) = {: 1.2e}'.format(tau_vs_opto_comp_summary['stats']['chi2_area']['p']),transform=ax[0].transAxes)
            ax[0].text(1.5,-.4,'p(tau_stim) = {: 1.2e}'.format(tau_vs_opto_comp_summary['stats']['chi2_tau_stimd']['p']),transform=ax[0].transAxes)
            ax[0].text(1.5,-.3,'p(tau_nonstim) = {: 1.2e}'.format(tau_vs_opto_comp_summary['stats']['chi2_tau_nonstimd']['p']),transform=ax[0].transAxes)   
             
        # single-area case
        else:
            # handle axes
            if axis_handles is None:
                fig = plt.figure()
                ax  = plt.gca()
            else:
                ax  = axis_handles
                fig = ax.get_figure()
                
            plt.colorbar(ax.imshow(analysis_results_v1['resp_prob_by_tau'],aspect='auto',cmap='bone',label='Response probability'),label='Response probability')
            ax.set_title(params['general_params'][area+'_lbl'])
            ax.set_xticks([0,1])
            ax.set_yticks([0,1])
            ax.set_xticklabels(['short $\\tau$','long $\\tau$'])
            ax.set_yticklabels(['short $\\tau$','long $\\tau$'])
            ax.set_xlabel('Responding neurons')
            ax.set_ylabel("Stim'd neurons")
        
    # plot response probability by tau over time
    elif plot_what == 'prob_by_time' or plot_what == 'prob_by_time_by_overall': 
        num_plots = len(analysis_results_v1['resp_prob_by_tau_over_time'])
        time_lbls = analysis_results_v1['tau_over_time_axis_sec']
        
        # multi-area case
        if area is None:
             # handle axes
            if axis_handles is None:
                fig = plt.figure()
                ax  = list()
                for iPlot in range(num_plots):
                    ax.append(plt.subplot(2,num_plots,iPlot+1))
                for iPlot in range(num_plots):
                    ax.append(plt.subplot(2,num_plots,iPlot+1+num_plots))
                    
            else:
                if len(axis_handles) != 2*len(analysis_results_v1['resp_prob_by_tau_over_time']):
                    print('need two axis handles for this plot, returning just the stats')
                    return None, None, tau_vs_opto_comp_summary
                ax  = axis_handles
                fig = axis_handles[0].get_figure()
                
            # choose what to plot
            if plot_what == 'prob_by_time':
                v1_mat = analysis_results_v1['resp_prob_by_tau_over_time']
                m2_mat = analysis_results_m2['resp_prob_by_tau_over_time']
            else:
                v1_mat = analysis_results_v1['resp_prob_by_tau_over_time_by_overall']
                m2_mat = analysis_results_m2['resp_prob_by_tau_over_time_by_overall']
                
            # put everyone on the same scale
            this_max = list()
            this_min = list()
            for iTime in range(num_plots):
                probs = np.concatenate((v1_mat[iTime].flatten(),m2_mat[iTime].flatten()))
                this_max.append(np.max(probs))
                this_min.append(np.min(probs))
            vmin = np.min(this_min)
            vmax = np.max(this_max)
            
            # plot heatmaps over time
            for iTime in range(num_plots):
                ax[iTime].imshow(v1_mat[iTime],aspect='auto',cmap='bone',vmax=vmax,vmin=vmin)
                ax[iTime].set_title(params['general_params']['V1_lbl']+str(time_lbls[iTime]))
                ax[iTime+num_plots].imshow(m2_mat[iTime],aspect='auto',cmap='bone',vmax=vmax,vmin=vmin)
                ax[iTime+num_plots].set_title(params['general_params']['M2_lbl']+str(time_lbls[iTime]))
                
                ax[iTime].set_xticks([0,1])
                ax[iTime].set_yticks([0,1])
                ax[iTime+num_plots].set_xticks([0,1])
                ax[iTime+num_plots].set_yticks([0,1])
                
                if iTime == num_plots-1:
                    plt.colorbar(ax[iTime].imshow(m2_mat[iTime],aspect='auto',cmap='bone',vmax=vmax,vmin=vmin,label='Response prob.'))
                
                if iTime == 0:
                    ax[iTime].set_xticklabels(['short $\\tau$','long $\\tau$'])
                    ax[iTime].set_yticklabels(['short $\\tau$','long $\\tau$'])
                    ax[iTime+num_plots].set_yticklabels(['short $\\tau$','long $\\tau$'])
                    ax[iTime+num_plots].set_xticklabels(['short $\\tau$','long $\\tau$'])
                    ax[iTime].set_ylabel("Stim'd neurons")
                else:
                    ax[iTime].set_xticklabels(['',''])
                    ax[iTime].set_yticklabels(['',''])
                    ax[iTime+num_plots].set_xticklabels(['',''])
                    ax[iTime+num_plots].set_yticklabels(['',''])
                if iTime == np.round(num_plots/2)-1:
                    ax[iTime+num_plots].set_xlabel('Responding neurons')
            
        # single-area case            
        else:
            if axis_handles is None:
                fig = plt.figure()
                ax  = list()
                for iPlot in range(num_plots):
                    ax.append(plt.subplot(1,num_plots,iPlot+1))
            else:
                ax  = axis_handles
                fig = axis_handles[0].get_figure()    
            
            # choose what to plot
            if plot_what == 'prob_by_time':
                if area == 'V1':
                    mat = analysis_results_v1['resp_prob_by_tau_over_time']
                else:
                    mat = analysis_results_m2['resp_prob_by_tau_over_time']
            else:
                if area == 'V1':
                    mat = analysis_results_v1['resp_prob_by_tau_over_time_by_overall']
                else:
                    mat = analysis_results_m2['resp_prob_by_tau_over_time_by_overall']
                
            # put everyone on the same scale
            this_max = list()
            this_min = list()
            for iTime in range(num_plots):
                probs = mat[iTime].flatten()
                this_max.append(np.max(probs))
                this_min.append(np.min(probs))
            vmin = np.min(this_min)
            vmax = np.max(this_max)
            
            # plot heatmaps over time
            for iTime in range(num_plots):
                ax[iTime].imshow(mat[iTime],aspect='auto',cmap='bone',vmax=vmax,vmin=vmin)
                ax[iTime].set_title(params['general_params'][area+'_lbl']+str(time_lbls[iTime]))
                
                ax[iTime].set_xticks([0,1])
                ax[iTime].set_yticks([0,1])
            
                if iTime == num_plots-1:
                    plt.colorbar(ax[iTime].imshow(mat[iTime],aspect='auto',cmap='bone',vmax=vmax,vmin=vmin,label='Response prob.'))
                
                if iTime == 0:
                    ax[iTime].set_xticklabels(['short $\\tau$','long $\\tau$'])
                    ax[iTime].set_yticklabels(['short $\\tau$','long $\\tau$'])
                    ax[iTime].set_ylabel("Stim'd neurons")
                else:
                    ax[iTime].set_xticklabels(['',''])
                    ax[iTime].set_yticklabels(['',''])
                if iTime == np.round(num_plots/2)-1:
                    ax[iTime].set_xlabel('Responding neurons')
        
    # plot peak time or magnitude by tau
    elif plot_what == 'peak_time' or plot_what == 'peak_mag': 
        
        if axis_handles is None:
            fig = plt.figure()
            ax  = fig.gca()
                
        # multi-area case
        if area is None:
            if plot_what == 'peak_time':
                yvals_v1 = analysis_results_v1['peak_time_by_tau_mean']
                sem_v1   = analysis_results_v1['peak_time_by_tau_sem']
                yvals_m2 = analysis_results_m2['peak_time_by_tau_mean']
                sem_m2   = analysis_results_m2['peak_time_by_tau_sem']
                lbl      = 'Peak time (sec)'
                stats    = tau_vs_opto_comp_summary['stats']['anova_peak_time_by_tau']
                texty    = 1.5
            else:
                yvals_v1 = analysis_results_v1['peak_mag_by_tau_mean']
                sem_v1   = analysis_results_v1['peak_mag_by_tau_sem']
                yvals_m2 = analysis_results_m2['peak_mag_by_tau_mean']
                sem_m2   = analysis_results_m2['peak_mag_by_tau_sem']
                lbl      = 'Peak magnitude (z-score)'
                stats    = tau_vs_opto_comp_summary['stats']['anova_peak_mag_by_tau']
                texty    = 2
            
            ax.errorbar(analysis_results_v1['tau_xaxis_sec'],yvals_v1,yerr=sem_v1,color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'])
            ax.errorbar(analysis_results_m2['tau_xaxis_sec'],yvals_m2,yerr=sem_m2,color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'])
            ax.set_xlabel('$\\tau$ (sec)')
            ax.set_ylabel(lbl)
            ax.legend()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # print pvals
            pval_area     = stats['p-unc'].loc[0]
            pval_tau      = stats['p-unc'].loc[1]
            pval_interact = stats['p-unc'].loc[2]
            ax.text(.5,texty,'p(area) = {:1.2g}'.format(pval_area)) 
            ax.text(.5,texty*.95,'p(tau) = {:1.2g}'.format(pval_tau)) 
            ax.text(.5,texty*.9,'p(area*tau) = {:1.2g}'.format(pval_interact)) 
        
        # single-area case    
        else:
            if plot_what == 'peak_time':
                if area == 'V1':
                    xvals = analysis_results_v1['tau_xaxis_sec']
                    yvals = analysis_results_v1['peak_time_by_tau_mean']
                    sem   = analysis_results_v1['peak_time_by_tau_sem']
                    lbl   = 'Peak time (sec)'
                    cl    = params['general_params']['V1_cl']
                else:
                    xvals = analysis_results_m2['tau_xaxis_sec']
                    yvals = analysis_results_m2['peak_time_by_tau_mean']
                    sem   = analysis_results_m2['peak_time_by_tau_sem']
                    lbl   = 'Peak time (sec)'
                    cl    = params['general_params']['M2_cl']
            else:
                if area == 'V1':
                    xvals = analysis_results_v1['tau_xaxis_sec']
                    yvals = analysis_results_v1['peak_mag_by_tau_mean']
                    sem   = analysis_results_v1['peak_mag_by_tau_sem']
                    lbl   = 'Peak magnitude (z-score)'
                    cl    = params['general_params']['V1_cl']
                else:
                    xvals = analysis_results_m2['tau_xaxis_sec']
                    yvals = analysis_results_m2['peak_mag_by_tau_mean']
                    sem   = analysis_results_m2['peak_mag_by_tau_sem']
                    lbl   = 'Peak magnitude (z-score)'
                    cl    = params['general_params']['M2_cl']
                    
            ax.errorbar(xvals,yvals,yerr=sem,color=cl)
            ax.set_xlabel('$\\tau$ (sec)')
            ax.set_ylabel(lbl)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
    else:
        print('unknown plot_what parameter, returning nothing')
        return None, None, None
    
    return ax, fig, tau_vs_opto_comp_summary
                    
# ---------------
# %% plot taus on FOV
def plot_resp_fov(area, which_sess=0, which_stim=0, expt_type='standard', resp_type='dff', plot_what='peak_mag', prctile_cap=[0,98], signif_only=False, highlight_signif=True, axis_handle=None):

    """
    plot_resp_fov(area, which_sess=0, which_stim=0, expt_type='standard', resp_type='dff', plot_what='peak_mag', prctile_cap=[0,98], signif_only=False, highlight_signif=True, axis_handle=None)
    plots response properties on FOV
    
    INPUTS:
        area             : str, 'V1' or 'M2'
        which_sess       : int, session id (default is 0)
        which_stim       : int, stim id (default is 0)
        expt_type        : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type        : str, 'dff' (default) or 'deconv'
        plot_what        : str, 'peak_mag' (default), 'peak_time', 'full_seq' (sequence of frames at resolution set by params['response_time_bins'])
        prctile_cap      : list, [0,98] (default), percentile cap for colormap
        signif_only      : bool, True (default), only plot significant neurons
        highlight_signif : bool, True (default), highlight significant neurons 
        axis_handle      : axis handle for plotting (optional)
    
    OUTPUTS:
        ax   : axis handle of plot
        fig  : figure handle of plot
    """
    
    # fetch response stats
    if plot_what == 'full_seq': # for plotting a sequence of frames
        response_stats = get_avg_trig_responses(area, params=params, expt_type=expt_type, resp_type=resp_type, eg_ids=which_sess, signif_only=signif_only, which_neurons='all', as_matrix=True)
        stim_idx = np.argwhere(response_stats['stim_ids']==which_stim).flatten()
        all_vals = np.array(response_stats['trig_dff_avgs'])[stim_idx].tolist()
        taxis    = response_stats['time_axis_sec']
        is_sig   = np.argwhere(response_stats['is_sig'][stim_idx]==1).flatten()

        # bin responses to desired resolution
        tbins      = params['fov_seq_time_bins']
        num_frames = len(tbins)-1
        num_cells  = len(all_vals)
        fvals      = np.zeros((num_cells,num_frames))
        faxis      = tbins[:-1]+(np.diff(tbins.reshape((np.size(tbins),1))[:,0])[0]/2)
        for iBin in range(num_frames):
            idx = np.argwhere(np.logical_and(taxis>tbins[iBin],taxis<=tbins[iBin+1])).flatten()
            for iCell in range(num_cells):
                fvals[iCell,iBin] = np.nanmean(np.array(all_vals[iCell])[idx])
        
        # define list of axes
        if axis_handle is None:
            ax  = list()
        else:
            ax  = axis_handle
        fig = None
        lbl = '$\\Delta$F/F (z-score)'
        
    else: # single frames (peak time or magnitude)
        response_stats = get_full_resp_stats(area, params=params, expt_type=expt_type, resp_type=resp_type, eg_ids=which_sess, which_neurons='all')
        stim_idx       = np.argwhere(response_stats['stim_ids']==which_stim).flatten()
        
        if plot_what == 'peak_mag':
            vals = np.array(response_stats['max_or_min_vals'])[stim_idx]
            lbl  = 'Peak $\\Delta$F/F (z-score)'
        elif plot_what == 'peak_time':
            vals = np.array(response_stats['max_or_min_times_sec'])[stim_idx]
            lbl  = 'Peak time (sec)'
        else:
            print('unknown plot_what, returning nothing')
            return None, None
        
        # make non-significant responses == 0
        is_sig = np.argwhere(response_stats['is_sig'][stim_idx]==1).flatten()
        if signif_only:
            is_not_sig       = np.argwhere(response_stats['is_sig'][stim_idx]==0).flatten()
            vals[is_not_sig] = 0
                
        if axis_handle is None:
            fig = plt.figure()
            ax  = plt.gca()
        else:
            ax  = axis_handle
            fig = axis_handle.get_figure() 
    
    # fetch roi coordinates for desired session
    start_time = time.time()
    print('Fetching roi coordinates...')
    roi_keys   = np.array(response_stats['roi_keys'])[stim_idx].tolist()
    row_pxls, col_pxls           = (VM['twophoton'].Roi2P & roi_keys).fetch('row_pxls','col_pxls')
    num_rows,num_cols,um_per_pxl = (VM['twophoton'].Scan & roi_keys[0]).fetch1('lines_per_frame','pixels_per_line','microns_per_pxl_y')
    roi_coords = [row_pxls,col_pxls]
    im_size    = (num_rows,num_cols)
    stimd_idx  = np.argwhere(response_stats['is_stimd'][stim_idx]).flatten().tolist()
    print('     done after {:1.2f} sec'.format(time.time()-start_time))
    
    # send to generic function
    if plot_what == 'full_seq':
        if isinstance(stimd_idx, list) == False:
            stimd_idx = [stimd_idx]
        if len(ax) == 0:
            fig, ax = plt.subplots(1,num_frames)
        imax = np.percentile(np.abs(fvals),prctile_cap[1])
        for iF in range(num_frames):
            if iF == num_frames-1:
                cbar = True
            else:
                cbar = False
                
            iax, fig = plot_fov_heatmap(roi_vals=fvals[:,iF].tolist(), roi_coords=roi_coords, im_size=im_size, um_per_pxl=um_per_pxl, \
                                        prctile_cap=prctile_cap, cbar_lbl=lbl, axisHandle=ax[iF], figHandle=fig, \
                                        cmap='coolwarm', background_cl = 'grey', plot_colorbar=cbar, max_min=[-imax,imax])
            
            iax.set_title('{} sec'.format(faxis[iF]))
            
            # add arrow on stim'd neuron
            for istim in stimd_idx:
                x = np.median(roi_coords[1][istim])
                y = np.median(roi_coords[0][istim]) - 15
                iax.plot(x,y,'kv',ms=4)
            
            # draw a circle around significant neurons 
            if highlight_signif:
                for isig in is_sig.tolist():
                    if isig in stimd_idx:
                        continue
                    x = np.median(roi_coords[1][isig]) + 8
                    y = np.median(roi_coords[0][isig]) + 14
                    iax.text(x,y,'*',color='k',fontsize=8)
                
            ax[iF] = iax
    else:
        if plot_what == 'peak_mag':
            imax      = np.percentile(np.abs(vals),prctile_cap[1])
            imin      = -imax
            maxmin    = [imin,imax]
            cmap_name = 'coolwarm'
        else:
            maxmin    = None
            cmap_name = 'viridis'
            
        ax, fig = plot_fov_heatmap(roi_vals=vals.tolist(), roi_coords=roi_coords, im_size=im_size, um_per_pxl=um_per_pxl, \
                                    prctile_cap=prctile_cap, cbar_lbl=lbl, axisHandle=ax, figHandle=fig, \
                                    cmap=cmap_name, background_cl = 'grey',max_min=maxmin)

        # add arrow on stim'd neuron
        for istim in stimd_idx:
            x = np.median(roi_coords[1][istim])
            y = np.median(roi_coords[0][istim]) - 15
            ax.plot(x,y,'kv',ms=6)
        
        # draw a circle around significant neurons 
        if highlight_signif:
            for isig in is_sig.tolist():
                if isig in stimd_idx:
                    continue
                x = np.median(roi_coords[1][isig]) + 8
                y = np.median(roi_coords[0][isig]) + 14
                ax.text(x,y,'*',color='k',fontsize=12)
        
    if fig is None:
        fig = axis_handle.get_figure() 
        
    return ax, fig

# ---------------
# %% PCA of a single baseline session
def baseline_sess_pca(expt_key,params=params,resp_type='dff'):
    
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
    
    # get just included, non-stimd rois
    sess_key = {'subject_fullname': expt_key['subject_fullname'],
                'session_date'    : expt_key['session_date'],
                'trigdff_param_set_id': params['trigdff_param_set_id_{}'.format(resp_type)],
                'trigdff_inclusion_param_set_id': params['trigdff_inclusion_param_set_id'],
                }

    incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & sess_key).fetch('is_included', 'KEY')
    stimd      = (twop_opto_analysis.TrigDffTrialAvg & keys).fetch('is_stimd')
    taxis      = (twop_opto_analysis.TrigDffTrialAvg & keys[0]).fetch1('time_axis_sec')
    idx        = np.argwhere(np.logical_and(np.array(incl)==1,np.array(stimd)==0)).flatten()
    keys       = list(np.array(keys)[idx])

    # fetch their dffs and restrict to baseline
    dff          = (VM['twophoton'].Dff2P & keys).fetch('dff')
    total_frames = np.size(dff[0])
    baseline_idx = spont_timescales.get_baseline_frame_idx(sess_key,total_frames)
    frame_int    = np.diff(taxis)[0]
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

# ---------------
# %% project trial responses onto PCA components
def project_trial_responses(expt_key, params=params, resp_type='dff'):
    
    """
    project_trial_responses(expt_key, params=params, resp_type='dff')
    projects trial responses onto PCA components
    
    INPUTS:
        expt_key      : dict, session key
        params        : dict, analysis parameters
        resp_type     : str, 'dff' (default) or 'deconv'
    
    OUTPUTS:
        trial_proj_results : dict, projection results
    """
    
    # get PCA results
    pca_results = baseline_sess_pca(expt_key,params=params,resp_type=resp_type)
    keys        = pca_results['roi_keys']
    pca_obj     = pca_results['pca_obj']
    dff_means   = pca_results['dff_means']
    dff_stds    = pca_results['dff_stds']
    smooth_win  = pca_results['smooth_win_samp']
    
    # get trial responses for this session
    start_time = time.time()
    print('     Fetching trial responses... (takes a while)')
    trial_ids, trials, stim_ids, roi_ids = (twop_opto_analysis.TrigDffTrial & keys).fetch('trial_id', 'trig_dff', 'stim_id', 'roi_id')
    taxis = (twop_opto_analysis.TrigDffTrialAvg & keys[0]).fetch1('time_axis_sec')
    print('     done after {:1.2f} min'.format((time.time()-start_time)/60))
    
    # project trial responses onto PCA components
    # do it separately for baseline and respose to avoid NaNs (from PMT shuttering)
    basel_idx = np.argwhere(np.logical_and(taxis > -params['pca_basel_sec'], taxis < -0.25)).flatten()
    first_idx = np.argwhere(np.isnan(trials[0])).flatten()[-1]+10
    last_idx  = np.argwhere(taxis<= params['pca_resp_sec']).flatten()[-1]
    resp_idx  = np.arange(first_idx,last_idx+1)
    
    stim_ids     = np.array(stim_ids).flatten()
    trial_ids    = np.array(trial_ids).flatten()
    unique_stims = np.unique(stim_ids).tolist()
    unique_rois  = np.unique(roi_ids).tolist()
    num_cells    = len(unique_rois)
    trial_projs_basel = [None]*len(unique_stims)
    trial_projs_resp  = [None]*len(unique_stims)
    
    for iStim, stim in enumerate(unique_stims):
        sidx          = stim_ids==stim
        unique_trials = np.unique(trial_ids[sidx]).tolist()
        trial_projs_basel[iStim] = [None]*len(unique_trials)
        trial_projs_resp[iStim]  = [None]*len(unique_trials)
        for iTrial, trial in enumerate(unique_trials):
            
            # select trials
            tidx      = np.logical_and(trial_ids==trial,sidx)
            trial_mat = np.zeros((len(taxis),num_cells))
            
            # build matrix
            for iCell, roi in enumerate(unique_rois):
                cidx = np.argwhere(np.logical_and(tidx,np.array(roi_ids)==roi)).flatten()
                if len(cidx)==0:
                    trial_mat[:,iCell] = np.zeros(len(taxis)) 
                else:
                    dff = trials[cidx[0]]
                    
                    # z score to baseline mean / std; smooth
                    dff = (dff - dff_means[iCell]) / dff_stds[iCell]
                    dff = general_stats.moving_average(dff,num_points=smooth_win)
                    trial_mat[:,iCell] = dff
            
            # project onto PCA components
            trial_projs_basel[iStim][iTrial] = pca_obj.transform(trial_mat[basel_idx,:])
            trial_projs_resp[iStim][iTrial]  = pca_obj.transform(trial_mat[resp_idx,:])
    
    # compute pairwise distances between trials (baseline and post-stim)
    num_pcs    = params['pca_num_components']
    basel_dist = list()
    resp_dist  = list()
    for iStim in range(len(unique_stims)):
        num_trials = len(trial_projs_basel[iStim])
        for iTrial1 in range(num_trials):
            for iTrial2 in range(num_trials):
                if iTrial1 > iTrial2:
                    dist = np.linalg.norm(trial_projs_basel[iStim][iTrial1][:num_pcs]-trial_projs_basel[iStim][iTrial2][:num_pcs])
                    basel_dist.append(dist)
                    
                    dist = np.linalg.norm(trial_projs_resp[iStim][iTrial1][:num_pcs]-trial_projs_resp[iStim][iTrial2][:num_pcs])
                    resp_dist.append(dist)

    # correlation between distances
    basel_dist = np.array(basel_dist).flatten()
    resp_dist  = np.array(resp_dist).flatten()
    corr_basel_vs_resp, p_basel_vs_resp = scipy.stats.pearsonr(basel_dist,resp_dist)
    
    # collect results and return 
    trial_proj_results = {
                        'pca_results'       : pca_results,
                        'num_pcs'           : num_pcs,
                        'var_explained'     : pca_results['cum_var_explained'][num_pcs],
                        'params'            : deepcopy(params),
                        'trial_projs_basel' : trial_projs_basel,
                        'trial_projs_resp'  : trial_projs_resp,
                        'basel_dists'       : basel_dist,
                        'resp_dists'        : resp_dist,
                        'p_basel_vs_resp'   : p_basel_vs_resp,
                        'corr_basel_vs_resp': corr_basel_vs_resp,
                        'num_stim'          : len(unique_stims)
                        }
    
    return trial_proj_results
    
# ---------------
# %% batch PCA trial projection, baseline vs. post-stim
def batch_trial_pca(area, params=params, expt_type='standard+high_trial_count', resp_type='dff', eg_ids=None):
    
    """
    batch_trial_pca(area, params=params, expt_type='standard+high_trial_count', resp_type='dff', eg_ids=None)
    runs batch analysis of comparing baseline and post-stim pca by calling project_trial_responses()
    
    INPUTS:
        area        : str, 'V1' or 'M2'
        params      : dict, analysis parameters (default is params from top of this script)
        expt_type   : str, 'standard+high_trial_count' (default), 'standard', 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type   : str, 'dff' (default) or 'deconv'
        eg_ids      : list of int, experiment group ids to restrict to specific experiments (optional, default is None)
        
    OUTPUTS:
        trial_pca_results : dict with summary data of single-trial pca projections
    """
    
    start_time      = time.time()
    print('Performing batch trial PCA...')
    
    # get relevant keys
    if expt_type == 'standard+high_trial_count':
        expt_keys1 = get_keys_for_expt_types(area, params=params, expt_type='standard')
        expt_keys2 = get_keys_for_expt_types(area, params=params, expt_type='high_trial_count')
        expt_keys  = expt_keys1 + expt_keys2
    else:
        expt_keys  = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
        
    # restrict to only desired rec/stim if applicable
    if eg_ids is not None:
        if isinstance(eg_ids,list) == False:
            eg_ids = [eg_ids]
        expt_keys = list(np.array(expt_keys)[np.array(eg_ids).astype(int)])
        
    # loop through experiments and call method above
    single_expt_results = list()
    basel_dists = list()
    resp_dists  = list()
    ccs         = list()
    ps          = list()
    var_expl    = list()
    num_expts   = len(expt_keys)
    num_stim    = 0
    for ct, expt_key in enumerate(expt_keys):
        print('{} of {}'.format(ct+1,num_expts))
        these_results = project_trial_responses(expt_key, params=params, resp_type=resp_type)
        single_expt_results.append(these_results)
        [basel_dists.append(x) for x in these_results['basel_dists'].tolist()]
        [resp_dists.append(x) for x in these_results['resp_dists'].tolist()]
        ccs.append(these_results['corr_basel_vs_resp'])
        ps.append(these_results['p_basel_vs_resp'])
        var_expl.append(these_results['var_explained']) 
        num_stim += these_results['num_stim']
        
    # overall correlation, collect results
    basel_dists = np.array(basel_dists).flatten()
    resp_dists  = np.array(resp_dists).flatten()
    ccs         = np.array(ccs).flatten()
    ps          = np.array(ps).flatten()
    var_expl    = np.array(var_expl).flatten()
    corr_basel_vs_resp, p_basel_vs_resp = scipy.stats.pearsonr(basel_dists,resp_dists)
    
    trial_pca_results = {
                        'single_expt_results' : single_expt_results,
                        'num_pcs_used'        : params['pca_num_components'],
                        'var_explained'       : var_expl,
                        'num_sess'            : num_expts,
                        'total_stimd'         : num_stim, 
                        'basel_dists'         : basel_dists,
                        'resp_dists'          : resp_dists,
                        'p_basel_vs_resp'     : p_basel_vs_resp,
                        'corr_basel_vs_resp'  : corr_basel_vs_resp,
                        'p_by_expt'           : ps,
                        'corr_by_expt'        : ccs,
                        }
    
    print('done after {:1.2f} min'.format((time.time()-start_time)/60))
    
    return trial_pca_results

# ---------------
# %% plot scatter of baseline vs post-stim pca projections
def plot_pca_dist_scatter(area, params=params, expt_type='standard+high_trial_count', resp_type='dff', eg_ids=None, trial_pca_results=None, axis_handle=None):
    
    """
    plot_pca_dist_scatter(area, params=params, expt_type='standard+high_trial_count', resp_type='dff', eg_ids=None, trial_pca_results=None, axis_handle=None)
    runs batch analysis of comparing baseline and post-stim pca by calling project_trial_responses()
    
    INPUTS:
        area        : str, 'V1' or 'M2'
        params      : dict, analysis parameters (default is params from top of this script)
        expt_type   : str, 'standard+high_trial_count' (default), 'standard', 'short_stim', 'high_trial_count', 'multi_cell'
        resp_type   : str, 'dff' (default) or 'deconv'
        eg_ids      : list of int, experiment group ids to restrict to specific experiments (optional, default is None)
        
    OUTPUTS:
        trial_pca_results : dict with summary data of single-trial pca projections
    """
    
    # run analysis if necessary
    if trial_pca_results is None:
        trial_pca_results = batch_trial_pca(area=area, params=params, expt_type=expt_type, resp_type=resp_type, eg_ids=eg_ids)
        
    # axis handles
    if axis_handle is None:
        plt.figure()
        ax  = plt.gca()
    else:
        ax  = axis_handle
    
    # plot scatter      
    ax.plot(trial_pca_results['basel_dists'],trial_pca_results['resp_dists'],'o', \
        color=params['general_params'][area+'_sh'],mew=0,label=params['general_params'][area+'_lbl']+'trial pairs')
    yl = ax.get_ylim()
    xl = ax.get_xlim()
    ax.text(xl[0]*1.05,yl[1]*.95,'r = {:1.2f}'.format(trial_pca_results['corr_basel_vs_resp']))
    ax.text(xl[0]*1.05,yl[1]*.90,'p = {:1.2g}'.format(trial_pca_results['p_basel_vs_resp']))
    
    # plot regression line
    olsfit = sm.OLS(trial_pca_results['resp_dists'],sm.add_constant(trial_pca_results['basel_dists'])).fit()
    x_hat  = np.arange(xl[0]-5, xl[1]+6)
    y_hat  = olsfit.predict(sm.add_constant(x_hat))
    predci = olsfit.get_prediction(sm.add_constant(x_hat)).summary_frame()
    ci_up  = predci.loc[:,'mean_ci_upper']
    ci_low = predci.loc[:,'mean_ci_lower']
    ax.plot(x_hat,y_hat,'-',color=params['general_params'][area+'_cl'])
    ax.plot(x_hat,ci_up,'--',color=params['general_params'][area+'_cl'],lw=.4)
    ax.plot(x_hat,ci_low,'--',color=params['general_params'][area+'_cl'],lw=.4)
    
    ax.legend()
    ax.set_xlabel('Trial Euclidean dist. (baseline)')
    ax.set_ylabel('Trial Euclidean dist. (post-stim)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return ax, trial_pca_results
