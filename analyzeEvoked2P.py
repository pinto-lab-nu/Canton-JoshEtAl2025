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
from utils.plotting import plot_fov_heatmap_circle
from utils.plotting import plot_fov_heatmap_blur
from utils.plotting import highlight_rois
from analyzeSpont2P import params as tau_params
import analyzeSpont2P
import Canton_Josh_et_al_2025_analysis_plotting_functions as analysis_plotting_functions



import time

# connect to dj 
VM     = connect_to_dj.get_virtual_modules()

# %% declare default params, inherit general params from tau_params
params = {
        'random_seed'                    : 42, 
        'trigdff_param_set_id_dff'       : 2,      #Z-scored etc.
        'trigdff_param_set_id_deconv'    : 5, 
        'trigdff_inclusion_param_set_id' : 9,    #timing, distance,etc
        'trigdff_inclusion_param_set_id_notiming' : 3, # for some analyses (e.g. inhibition), we may want to relax trough timing constraint
        'trigspeed_param_set_id'         : 1,
        'prop_resp_bins'                 : np.arange(0,.41,.01),
        'dist_bins_resp_prob'            : np.arange(25,500,50),
        'response_magnitude_bins'        : np.arange(-2.5,7.75,.25),
        'tau_bins'                       : np.arange(0,3,.5),
        'tau_by_time_bins'               : np.arange(0,5,.5),
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
        'pca_num_components'             : 10, # number of PCA for trial projections
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
# def get_keys_for_expt_types(area, params=params, expt_type='standard'):
    
#     """
#     get_keys_for_expt_types(area, params=params, expt_type='standard')
#     retrieves dj keys for experiments of a certain type for a given area
    
#     INPUTS:
#         area      : str, 'V1' or 'M2'
#         params    : dict, analysis parameters (default is params from top of this script)
#         expt_type : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
    
#     OUTPUTS:
#         keys : list of dj keys for experiments of the desired type
#     """
    
#     # get primary keys for query
#     mice              = params['general_params']['{}_mice'.format(area)]
    
#     # get relavant keys 
#     keys = list()
#     for mouse in mice:
#         opto_data = (twop_opto_analysis.Opto2PSummary & {'subject_fullname': mouse}).fetch(as_dict=True)
#         opto_keys = (twop_opto_analysis.Opto2PSummary & {'subject_fullname': mouse}).fetch("KEY",as_dict=True)
#         spiral_data=(VM['twophoton'].Opto2P & opto_keys).fetch(as_dict=True)
        
#         for ct, this_sess in enumerate(opto_data):
#             has_multicell_stim = bool(np.sum(np.array(this_sess['num_cells_per_stim'])>1)>1)
#             max_num_trials     = np.max(np.array(this_sess['num_trials_per_stim']))
#             max_dur            = np.max(np.array(this_sess['dur_sec_per_stim']))
            
#             # spiral_size        = np.max(np.array(this_sess['stim_spiral_size_um']))
            
#             # choose experiments that match type
#             if expt_type == 'standard':
#                 if np.logical_and((not has_multicell_stim), np.logical_and(max_num_trials <= 10 , max_dur >= 0.2)):
#                     keys.append(opto_keys[ct])
#             elif expt_type == 'short_stim':
#                 if np.logical_and((not has_multicell_stim), np.logical_and(max_num_trials <= 10 , max_dur >= 0.2)):   #OKKKKKK This needs fixing long term used spiral size since there is issue with max_dur and 'trials_num_spirals' (need repetition)
#                     keys.append(opto_keys[ct])
#             elif expt_type == 'high_trial_count':
#                 if np.logical_and((not has_multicell_stim) , max_num_trials > 10):
#                     keys.append(opto_keys[ct])
#             elif expt_type == 'multi_cell':
#                 if has_multicell_stim:
#                     keys.append(opto_keys[ct])
#             else:
#                 print('Unknown experiment type, doing nothing')
#                 return None
    
#     return keys

# %% get list of dj keys for sessions of a certain type
# 'standard' is single neurons with <= 10 trials & 5 spirals, 'short_stim' is single neurons <= 10 trials & < 5 spirals, 
# 'high_trial_count' is single neurons with > 10 trials, 'multi_cell' has at least one group with mutiple stim'd neurons

def get_keys_for_expt_types(area, params=params, expt_type='standard'):
    """
    Retrieves dj keys for experiments of a certain type for a given area.
    """
    mice = params['general_params']['{}_mice'.format(area)]
    keys = []

    for mouse in mice:
        opto_data   = (twop_opto_analysis.Opto2PSummary & {'subject_fullname': mouse}).fetch(as_dict=True)
        opto_keys   = (twop_opto_analysis.Opto2PSummary & {'subject_fullname': mouse}).fetch("KEY", as_dict=True)
        # spiral_sizes = (VM['twophoton'].Opto2P & opto_keys).fetch('stim_spiral_size_um')
        spiral_repetitions = (VM['twophoton'].Opto2P & opto_keys).fetch('trials_repetitions')


        for this_sess, spiral_rep_array, key in zip(opto_data, spiral_repetitions, opto_keys):
            has_multicell_stim = np.sum(np.array(this_sess['num_cells_per_stim']) > 1) > 1
            max_num_trials     = np.max(np.array(this_sess['num_trials_per_stim']))
            max_dur            = np.max(np.array(this_sess['dur_sec_per_stim']))
            spiral_reps       = np.max(np.array(spiral_rep_array))

            if expt_type == 'standard':
                if not has_multicell_stim and max_num_trials <= 10 and max_dur >= 0.2 and spiral_reps > 1:   #OKKKKKK This needs fixing long term used spiral size since there is issue with max_dur and 'trials_num_spirals' (need repetition)
                    keys.append(key)
            elif expt_type == 'short_stim':
                if not has_multicell_stim and max_num_trials <= 10 and spiral_reps < 2:  #OKKKKKK This needs fixing long term used spiral size since there is issue with max_dur and 'trials_num_spirals' (need repetition)
                    keys.append(key)
            elif expt_type == 'high_trial_count':
                if not has_multicell_stim and max_num_trials > 10:
                    keys.append(key)
            elif expt_type == 'multi_cell':
                if has_multicell_stim:
                    keys.append(key)
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
        if  len(prop)>0:  
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
    maxmins    = list()
    
    for ct, ikey in enumerate(expt_keys):
        print('     {} of {}...'.format(ct+1,num_expt))
        this_key = {'subject_fullname' : ikey['subject_fullname'], 
                    'session_date': ikey['session_date'], 
                    'trigdff_param_set_id': trigdff_param_set_id, 
                    'trigdff_inclusion_param_set_id': trigdff_inclusion_param_set_id
                    }
        
        # for some downstream analyses we need to keep track of roi order, so separate by stim
        these_stim_ids = list((twop_opto_analysis.Opto2PSummary & this_key).fetch1('stim_ids'))
        # these_stim_ids2 = ((twop_opto_analysis.Opto2PSummary & this_key).fetch1('stim_ids'))

 
        for this_stim in these_stim_ids:
            this_key['stim_id'] = this_stim   #Theres was a bug here due to float
            sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')

            # do selecion at the fetching level for speed
            if which_neurons == 'stimd':
                # stimd neurons bypass inclusion criteria
                avgs, sems, ts, stimd, com, peak, sids, maxmin, nkeys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('trig_dff_avg', 'trig_dff_sem', 'time_axis_sec', 'is_stimd', 'center_of_mass_sec_overall', 'time_of_peak_sec_overall', 'stim_id', 'max_or_min_dff', 'KEY')
            else:
                sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')
                idx  = np.argwhere(np.array(incl)==1).flatten()
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx])
                
                if signif_only:
                    idx  = np.argwhere(np.array(sig)==1).flatten()
                    keys = list(np.array(keys)[idx])
                    sig  = list(np.array(sig)[idx]) 
                
                avgs, sems, ts, stimd, com, peak, sids, maxmin, nkeys = (twop_opto_analysis.TrigDffTrialAvg & keys & 'is_stimd=0').fetch('trig_dff_avg', 'trig_dff_sem', 'time_axis_sec', 'is_stimd', 'center_of_mass_sec_overall', 'time_of_peak_sec_overall', 'stim_id', 'max_or_min_dff', 'KEY')
                
                # get stimd neurons if desired (bypass inclusion because of distance criterion)
                if which_neurons == 'all':
                    avgsst, semsst, tsst, stimdst, comst, peakst, sidsst, maxminst, nkeysst  = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('trig_dff_avg', 'trig_dff_sem', 'time_axis_sec', 'is_stimd', 'center_of_mass_sec_overall', 'time_of_peak_sec_overall', 'stim_id', 'max_or_min_dff', 'KEY')
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
                    [maxmin.append(mm) for mm in maxminst]
                        
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
            [maxmins.append(mm) for mm in maxmin]
            
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
                    'analysis_params': deepcopy(params),
                    'max_or_min_vals': np.array(maxmins).flatten()
                    }
    
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    return summary_data

# ---------------
# %% get response stats (peak, dist from stim etc) for an area and experiment type
def get_full_resp_stats_new(area, params=params, expt_type='standard', resp_type='dff', eg_ids=None, which_neurons='non_stimd'):
    
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
            
            if which_neurons == 'all':
                # stimdst, comst, maxminst, peaktst, troughtst, distst, sidst, nkeysst = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')
                sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')
                # n    = len(stimd)
                # sig  = list(np.ones(n))
                
                idx  = np.argwhere(np.array(incl)==1).flatten()
                n    = np.size(idx)
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx])
                
                # is_soma = np.array((VM['twophoton'].Roi2P & keys).fetch('is_soma'))
                # idx  = np.argwhere(np.array(is_soma)==1).flatten()
                # n    = np.size(idx)
                # keys = list(np.array(keys)[idx])
                # sig  = list(np.array(sig)[idx])
                
                stimd, com, maxmin, peakt, trought, dist, sid, nkeys = (twop_opto_analysis.TrigDffTrialAvg & keys).fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')

            # do selecion at the fetching level for speed
            elif which_neurons == 'stimd':
                # stimd neurons bypass inclusion criteria
                stimd, com, maxmin, peakt, trought, dist, sid, nkeys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')
                n    = len(stimd)
                sig  = list(np.ones(n))
                
            elif which_neurons == 'non_stimd':          
                # get included non-stimulated neurons first 
                sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')
                idx  = np.argwhere(np.array(incl)==1).flatten()
                n    = np.size(idx)
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx])

                stimd, com, maxmin, peakt, trought, dist, sid, nkeys = (twop_opto_analysis.TrigDffTrialAvg & keys & 'is_stimd=0').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')
             
            elif which_neurons == 'all_old':          
                # get included non-stimulated neurons first 
                sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')
                idx  = np.argwhere(np.array(incl)==1).flatten()
                n    = np.size(idx)
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx])

                stimd, com, maxmin, peakt, trought, dist, sid, nkeys = (twop_opto_analysis.TrigDffTrialAvg & keys & 'is_stimd=0').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')
                
                # get stimd neurons if desired (bypass inclusion because of distance criterion)
                # if which_neurons == 'all_old':
                stimdst, comst, maxminst, peaktst, troughtst, distst, sidst, nkeysst = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')
                # stimdst, comst, maxminst, peaktst, troughtst, distst, sidst, nkeysst = (twop_opto_analysis.TrigDffTrialAvg & this_key).fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id', 'KEY')

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
            # is_soma  = np.array((VM['twophoton'].Roi2P & nkeys).fetch('is_soma'))

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
            if unique_sids>0:
                n           = n / len(unique_sids)
            else:
                n=0
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
    cell_keys       = list()
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
            [cell_keys.append(er)for er in nkeys]

            # response prob by dist per experiment
            dist        = np.array(dist).flatten()
            sig         = np.array(sig).flatten()
            unique_sids = np.unique(sid)
            if unique_sids>0:
                n           = n / len(unique_sids)
            else:
                n=0
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
                    'analysis_params'       : deepcopy(params),
                    'cell_keys'                 :cell_keys
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

# %%
def compute_peak_time_by_distance(response_stats, area='V1',signif_only=True):
    """
    Bin neurons by distance from stimulation and compute average peak response time and SEM for each bin.
    
    INPUTS:
        response_stats : dict, output from compare_response_stats
        area           : str, 'V1' or 'M2'
    
    OUTPUTS:
        mean_peak_times : np.array, mean peak response time for each distance bin
        sem_peak_times  : np.array, SEM of peak time per bin
        counts_per_bin  : np.array, number of neurons per bin
    """
    # Extract data
    dist_axis  = response_stats[f'{area}_stats']['dist_axis']  # bin centers
    bin_edges  = np.append(dist_axis - np.diff(dist_axis)[0]/2, dist_axis[-1] + np.diff(dist_axis)[-1]/2)
    peak_times = np.array(response_stats[f'{area}_stats']['max_or_min_times_sec'])
    distances  = np.array(response_stats[f'{area}_stats']['dist_from_stim_um'])
    
    if signif_only:
        idx = np.argwhere(response_stats[f'{area}_stats']['is_sig']==1).flatten()
        peak_times =peak_times[idx]
        distances  =distances[idx]
    # Initialize outputs
    mean_peak_times = np.full_like(dist_axis, np.nan, dtype=float)
    sem_peak_times  = np.full_like(dist_axis, np.nan, dtype=float)
    counts_per_bin  = np.zeros_like(dist_axis, dtype=int)

    # Bin neurons by distance and compute mean + SEM
    for i in range(len(dist_axis)):
        in_bin = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
        bin_peaks = peak_times[in_bin]
        if len(bin_peaks) > 0:
            mean_peak_times[i] = np.nanmean(bin_peaks)
            sem_peak_times[i]  = np.nanstd(bin_peaks) / np.sqrt(len(bin_peaks))
            counts_per_bin[i]  = len(bin_peaks)

    return mean_peak_times, sem_peak_times, counts_per_bin


# %% plot comparison of different response stats between areas
def plot_response_stats_comparison(params=params, expt_type='standard', resp_type='dff', which_neurons='non_stimd', response_stats=None, axis_handle=None, plot_what='response_magnitude', signif_only=True, overlay_non_sig=False,fig_size=(4,4)):
    
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
        plot_what      : str, 'response_magnitude' (default), 'response_time', 'prop_by_dist_vs_total' (i.e. overall prop.), 'prop_by_dist_of_sig' (i.e. prop. of significant),'time_by_dist_vs_total'
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
            
    elif plot_what == 'time_by_dist_vs_total':
        plt.figure(figsize=fig_size)
        ax = plt.gca()
        xaxis = response_stats['V1_stats']['dist_axis']
        v1_mean, v1_sem, _ = compute_peak_time_by_distance(response_stats, area='V1',signif_only=signif_only)
        m2_mean, m2_sem, _ = compute_peak_time_by_distance(response_stats, area='M2',signif_only=signif_only)
        
        ax.errorbar(x=xaxis, y=v1_mean, yerr=v1_sem, color=params['general_params']['V1_cl'], label=params['general_params']['V1_lbl'], marker='.')
        ax.errorbar(x=xaxis, y=m2_mean, yerr=m2_sem, color=params['general_params']['M2_cl'], label=params['general_params']['M2_lbl'], marker='.')
    
        xlbl = 'Dist. from stimd ($\\mu$m)'
        ylbl = 'Response peak time (s)'  
            
    else:
        print('unknown plot_what, plotting nothing')
        return response_stats, None
    
    # plot
    if axis_handle is None:
        plt.figure(figsize=fig_size)
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
        
    elif plot_what == 'time_by_dist_vs_total':
        xaxis = response_stats['V1_stats']['dist_axis']
        v1_mean, v1_sem, _ = compute_peak_time_by_distance(response_stats, area='V1')
        m2_mean, m2_sem, _ = compute_peak_time_by_distance(response_stats, area='M2')
    
        ax.errorbar(x=xaxis, y=v1_mean, yerr=v1_sem, color=params['general_params']['V1_cl'], label=params['general_params']['V1_lbl'], marker='.')
        ax.errorbar(x=xaxis, y=m2_mean, yerr=m2_sem, color=params['general_params']['M2_cl'], label=params['general_params']['M2_lbl'], marker='.')
    
        xlbl = 'Dist. from stimd ($\\mu$m)'
        ylbl = 'Response peak time (s)'    
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
# # %% get single-trial opto-triggered responses for an area and experiment type
# def get_single_trial_data(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', eg_ids=None, signif_only=True, which_neurons='non_stimd', relax_timing_criteria=params['xval_relax_timing_criteria']):
    
#     """
#     get_single_trial_data(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', eg_ids=None, signif_only=True, which_neurons='non_stimd', relax_timing_criteria=params['xval_relax_timing_criteria'])
#     retrieves single-trial opto-triggered responses for a given area and experiment type
    
#     INPUTS:
#         area                  : str, 'V1' or 'M2' (default is 'M2')
#         params                : dict, analysis parameters (default is params from top of this script)
#         expt_type             : str, 'standard' (default), 'short_stim', 'high_trial_count', 'multi_cell'
#         resp_type             : str, 'dff' (default) or 'deconv'
#         eg_ids                : list of int, experiment group ids to restrict to (optional, default is None)
#         signif_only           : bool, if True only include significant neurons (default is True)
#         which_neurons         : str, 'non_stimd' (default), 'all', 'stimd'
#         relax_timing_criteria : bool, if True relax timing criteria (default is in params['xval_relax_timing_criteria']). 
#                                 This will fetch from a different param set that doesn't require the timing criteria
        
#     OUTPUTS:
#         trial_data : dict with summary data and single-trial responses
#     """
    
#     start_time      = time.time()
#     print('Fetching opto-triggered trials...')
    
#     # get relevant keys
#     expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    
#     # restrict to only desired rec/stim if applicable
#     if eg_ids is not None:
#         if isinstance(eg_ids,list) == False:
#             eg_ids = [eg_ids]
#         expt_keys = list(np.array(expt_keys)[np.array(eg_ids).astype(int)])
    
#     # loop through keys to fetch the responses
#     trigdff_param_set_id = params['trigdff_param_set_id_{}'.format(resp_type)]
#     if relax_timing_criteria:
#         trigdff_inclusion_param_set_id = params['trigdff_inclusion_param_set_id_notiming']
#     else:
#         trigdff_inclusion_param_set_id = params['trigdff_inclusion_param_set_id']
    
#     trial_resps = list()
#     trial_ids   = list()
#     stim_ids    = list()
#     roi_ids     = list()
#     peak_ts     = list()
#     peak_amp    = list()
#     coms        = list()
#     t_axes      = list()
#     num_expt    = len(expt_keys)
#     for ct, ikey in enumerate(expt_keys):
#         print('     {} of {}...'.format(ct+1,num_expt))
#         this_key = {'subject_fullname' : ikey['subject_fullname'], 
#                     'session_date': ikey['session_date'], 
#                     'trigdff_param_set_id': trigdff_param_set_id, 
#                     'trigdff_inclusion_param_set_id': trigdff_inclusion_param_set_id
#                     }
        
#         # do selecion at the fetching level for speed
#         if which_neurons == 'stimd':
#             # stimd neurons bypass inclusion criteria
#             avg_keys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('KEY')
#             # tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (twop_opto_analysis.TrigDffTrial & avg_keys).fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id')
#             sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & avg_keys).fetch('is_significant', 'is_included', 'KEY')
            
#             if signif_only:
#                 idx  = np.argwhere(np.array(sig)==1).flatten()
#                 keys = list(np.array(keys)[idx])
#                 sig  = list(np.array(sig)[idx]) 
        
#             tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (twop_opto_analysis.TrigDffTrial & keys).fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id')

#         elif which_neurons == 'non_stimd':
#             avg_keys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=0').fetch('KEY')
#             sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & avg_keys).fetch('is_significant', 'is_included', 'KEY')
#             idx  = np.argwhere(np.array(incl)==1).flatten()   #need to comment this in, but worried about distance cutoff
#             keys = list(np.array(keys)[idx])
#             sig  = list(np.array(sig)[idx])
            
#             if signif_only:
#                 idx  = np.argwhere(np.array(sig)==1).flatten()
#                 keys = list(np.array(keys)[idx])
#                 sig  = list(np.array(sig)[idx]) 
            
#             tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (twop_opto_analysis.TrigDffTrial & keys).fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id')
#         else:
#             print('code not implemented for this category of which_neurons, returning nothing')
#             return None

#         # pick trough or peak time, whichever is higher magnitude
#         maxmin_t = list()
#         for iNeuron in range(len(trought)):
#             if maxmin[iNeuron] < 0:
#                 maxmin_t.append(trought[iNeuron])
#             else:
#                 maxmin_t.append(peakt[iNeuron])
                
#         # flatten lists
#         [trial_resps.append(trial) for trial in trials]
#         [trial_ids.append(int(tid)) for tid in tids]
#         [stim_ids.append(int(sid)) for sid in sids]
#         [roi_ids.append(int(rid+(ct*10000))) for rid in rids] # 10000 is arbitrary experiment increment to make roi_ids unique
#         [t_axes.append(t) for t in ts]
#         [coms.append(co) for co in com]
#         [peak_ts.append(pt) for pt in maxmin_t]
#         [peak_amp.append(pa) for pa in maxmin]
            
#     # convert to arrays for easy indexing, trial and time vectors remain lists
#     trial_ids = np.array(trial_ids)
#     stim_ids  = np.array(stim_ids)
#     roi_ids   = np.array(roi_ids)
#     coms      = np.array(coms)
#     peak_ts   = np.array(peak_ts)
#     peak_amp  = np.array(peak_amp)
        
#     # collect summary data   
#     trial_data = {
#                 'trig_dff_trials'         : trial_resps, 
#                 'trial_ids'               : trial_ids, 
#                 'time_axis_sec'           : t_axes,
#                 'signif_only'             : signif_only,
#                 'stim_ids'                : stim_ids, 
#                 'roi_ids'                 : roi_ids, 
#                 'com_sec'                 : coms, 
#                 'peak_or_trough_time_sec' : peak_ts, 
#                 'peak_amp_value'          : peak_amp,
#                 'relax_timing_criteria'   : relax_timing_criteria,
#                 'which_neurons'           : which_neurons,
#                 'response_type'           : resp_type, 
#                 'experiment_type'         : expt_type, 
#                 'analysis_params'         : deepcopy(params),
#                 'sig'                     : sig
#                 }
    
#     end_time = time.time()
#     print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
#     return trial_data



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
    peak_amp    = list()
    coms        = list()
    t_axes      = list()
    roi_keys_list      = list()
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
            # tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (twop_opto_analysis.TrigDffTrial & avg_keys).fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id')
            sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & avg_keys).fetch('is_significant', 'is_included', 'KEY')
            
            if signif_only:
                idx  = np.argwhere(np.array(sig)==1).flatten()
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx]) 
        
            tids, trials, ts, sids, com, maxmin, peakt, trought, rids,roi_keys = (twop_opto_analysis.TrigDffTrial & keys).fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id',
                                                                                                                                'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 
                                                                                                                                'time_of_trough_sec_poststim', 'roi_id','KEY')
            trial_resps.extend(trials)
            trial_ids.extend([int(t) for t in tids])
            stim_ids.extend([int(s) for s in sids])
            roi_keys_list.extend(roi_keys)
            roi_ids.extend([int(r + ct * 10000) for r in rids]) # 10000 is arbitrary experiment increment to make roi_ids unique
            t_axes.extend(ts)
            coms.extend(com)
            peak_ts.extend(peakt)
            peak_amp.extend(maxmin)

        elif which_neurons == 'non_stimd':
            
            
            # Get all stim_ids for this experiment key
            try:
                these_stim_ids = list((twop_opto_analysis.Opto2PSummary & this_key).fetch1('stim_ids'))
            except:
                continue
        
            for this_stim in these_stim_ids:
                this_key['stim_id'] = this_stim  # Set specific stim ID for filtering
        
                # Get inclusion flags and keys for this stim ID
                sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch(
                    'is_significant', 'is_included', 'KEY'
                )
        
                # Filter by inclusion
                incl = np.array(incl)
                sig = np.array(sig)
                keys = np.array(keys)
        
                valid_idx = np.where(incl == 1)[0]
                keys = keys[valid_idx]
                sig = sig[valid_idx]
        
                # Further filter by significance if requested
                if signif_only:
                    valid_idx = np.where(sig == 1)[0]
                    keys = keys[valid_idx]
                    sig = sig[valid_idx]
        
                if len(keys) == 0:
                    continue
        
                # Now fetch the actual single-trial data for only those (stim_id, roi_id) combinations
                try:
                    tids, trials, ts, sids, com, maxmin, peakt, trought, rids,roi_keys = (
                        twop_opto_analysis.TrigDffTrial & list(keys)
                    ).fetch(
                        'trial_id', 'trig_dff', 'time_axis_sec', 'stim_id',
                        'center_of_mass_sec_poststim', 'max_or_min_dff',
                        'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id','KEY'
                    )
                except Exception as e:
                    print(f"Fetch error for stim {this_stim}: {e}")
                    continue
        
                maxmin_t = [trought[i] if maxmin[i] < 0 else peakt[i] for i in range(len(trought))]
        
                trial_resps.extend(trials)
                trial_ids.extend([int(t) for t in tids])
                stim_ids.extend([int(s) for s in sids])
                roi_keys_list.extend(roi_keys)
                roi_ids.extend([int(r + ct * 10000) for r in rids]) # 10000 is arbitrary experiment increment to make roi_ids unique
                t_axes.extend(ts)
                coms.extend(com)
                peak_ts.extend(maxmin_t)
                peak_amp.extend(maxmin)
                # sig_all.extend(sig.tolist())



        # elif which_neurons == 'non_stimd':
        #     avg_keys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=0').fetch('KEY')
        #     sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & avg_keys).fetch('is_significant', 'is_included', 'KEY')
        #     idx  = np.argwhere(np.array(incl)==1).flatten()   #need to comment this in, but worried about distance cutoff
        #     keys = list(np.array(keys)[idx])
        #     sig  = list(np.array(sig)[idx])
            
        #     if signif_only:
        #         idx  = np.argwhere(np.array(sig)==1).flatten()
        #         keys = list(np.array(keys)[idx])
        #         sig  = list(np.array(sig)[idx]) 
            
        #     tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (twop_opto_analysis.TrigDffTrial & keys).fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id')
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
        # [trial_resps.append(trial) for trial in trials]
        # [trial_ids.append(int(tid)) for tid in tids]
        # [stim_ids.append(int(sid)) for sid in sids]
        # [roi_ids.append(int(rid+(ct*10000))) for rid in rids] # 10000 is arbitrary experiment increment to make roi_ids unique
        # [t_axes.append(t) for t in ts]
        # [coms.append(co) for co in com]
        # [peak_ts.append(pt) for pt in maxmin_t]
        # [peak_amp.append(pa) for pa in maxmin]
            
    # convert to arrays for easy indexing, trial and time vectors remain lists
    trial_ids = np.array(trial_ids)
    stim_ids  = np.array(stim_ids)
    roi_ids   = np.array(roi_ids)
    coms      = np.array(coms)
    peak_ts   = np.array(peak_ts)
    peak_amp  = np.array(peak_amp)
        
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
                'peak_amp_value'          : peak_amp,
                'relax_timing_criteria'   : relax_timing_criteria,
                'which_neurons'           : which_neurons,
                'response_type'           : resp_type, 
                'experiment_type'         : expt_type, 
                'analysis_params'         : deepcopy(params),
                'sig'                     : sig,
                'roi_keys'                : roi_keys_list
                }
    
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    return trial_data
# %%
import time
import numpy as np
from copy import deepcopy

def get_single_trial_data_from_individual_session_stim(params,session_key, stim_id,area='M2', expt_type='standard', resp_type='dff', signif_only=False, which_neurons='non_stimd', relax_timing_criteria=False):
    """m
    fetch_single_trial_from_session(session_key, stim_id, params, resp_type='dff', signif_only=True, which_neurons='non_stimd', relax_timing_criteria=False)
    
    Fetches a single opto-triggered trial from a specific session and stim_id.

    INPUTS:
        session_key            : dict, must contain 'subject_fullname' and 'session_date'
        stim_id                : int, the specific stimulation ID to fetch
        params                 : dict, parameter dictionary
        resp_type              : str, 'dff' (default) or 'deconv'
        signif_only            : bool, include only significant neurons (default True)
        which_neurons          : str, one of 'non_stimd' (default), 'stimd'
        relax_timing_criteria  : bool, whether to use relaxed inclusion param set (default False)

    OUTPUT:
        trial_data : dict with keys:
            - 'trig_dff_trial'
            - 'trial_id'
            - 'time_axis_sec'
            - 'stim_id'
            - 'roi_id'
            - 'center_of_mass_sec_poststim'
            - 'peak_or_trough_time_sec'
            - 'peak_amp_value'
    """


    start_time = time.time()
    print("Fetching single-trial response...")
    
    trial_resps = list()
    trial_ids   = list()
    stim_ids    = list()
    roi_ids     = list()
    peak_ts     = list()
    peak_amp    = list()
    coms        = list()
    t_axes      = list()
    
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)

    this_key = {
        'subject_fullname': expt_keys[session_key]['subject_fullname'],
        'session_date': expt_keys[session_key]['session_date'],
        'stim_id': stim_id,
        'trigdff_param_set_id': params[f'trigdff_param_set_id_{resp_type}'],
        'trigdff_inclusion_param_set_id': params['trigdff_inclusion_param_set_id_notiming'] if relax_timing_criteria else params['trigdff_inclusion_param_set_id']
    }

    try:
        if which_neurons == 'stimd':
            avg_keys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('KEY')
        elif which_neurons == 'non_stimd':
            avg_keys = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=0').fetch('KEY')
        else:
            print("Invalid neuron type requested.")
            return None
    except:
        print("No matching average keys found.")
        return None

    try:
        sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & avg_keys).fetch('is_significant', 'is_included', 'KEY')
    except:
        print("No inclusion data found.")
        return None

    incl = np.array(incl)
    sig = np.array(sig)
    keys = np.array(keys)

    valid_idx = np.where(incl == 1)[0]
    keys = keys[valid_idx]
    sig = sig[valid_idx]

    if signif_only:
        valid_idx = np.where(sig == 1)[0]
        keys = keys[valid_idx]
        sig = sig[valid_idx]

    if len(keys) == 0:
        print("No valid trials found.")
        return None

    # Select only the first valid trial
    selected_key = keys[0]

    try:
        tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (
            twop_opto_analysis.TrigDffTrial & list(keys)
        ).fetch(
            'trial_id', 'trig_dff', 'time_axis_sec', 'stim_id',
            'center_of_mass_sec_poststim', 'max_or_min_dff',
            'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id'
        )
    except Exception as e:
        print(f"Fetch error for stim : {e}")

    maxmin_t = [trought[i] if maxmin[i] < 0 else peakt[i] for i in range(len(trought))]

    # trial_resps.extend(trials)
    # trial_ids.extend([int(t) for t in tids])
    # stim_ids.extend([int(s) for s in sids])
    # roi_ids.extend([int(r + ct * 10000) for r in rids])
    # t_axes.extend(ts)
    # coms.extend(com)
    # peak_ts.extend(maxmin_t)
    # peak_amp.extend(maxmin)
    # # sig_all.extend(sig.tolist())

   
    # [trial_resps.append(trial) for trial in trials]
    # [trial_ids.append(int(tid)) for tid in tids]
    # [stim_ids.append(int(sid)) for sid in sids]
    # [roi_ids.append(int(rid+(ct*10000))) for rid in rids] # 10000 is arbitrary experiment increment to make roi_ids unique
    # [t_axes.append(t) for t in ts]
    # [coms.append(co) for co in com]
    # [peak_ts.append(pt) for pt in maxmin_t]
    # [peak_amp.append(pa) for pa in maxmin]
        
    # convert to arrays for easy indexing, trial and time vectors remain lists
    trial_ids = np.array(tids)
    stim_ids  = np.array(stim_ids)
    roi_ids   = np.array(rids)
    coms      = np.array(com)
    peak_ts   = np.array(peakt)
    peak_amp  = np.array(maxmin_t)
        
    # collect summary data   
    trial_data = {
                'trig_dff_trials'         : trials, 
                'trial_ids'               : trial_ids, 
                'time_axis_sec'           : t_axes,
                'signif_only'             : signif_only,
                'stim_ids'                : stim_ids, 
                'roi_ids'                 : roi_ids, 
                'com_sec'                 : coms, 
                'peak_or_trough_time_sec' : peak_ts, 
                'peak_amp_value'          : peak_amp,
                'relax_timing_criteria'   : relax_timing_criteria,
                'which_neurons'           : which_neurons,
                'response_type'           : resp_type, 
                'experiment_type'         : expt_type, 
                'analysis_params'         : deepcopy(params),
                'sig'                     : sig
                }


    end_time = time.time()
    print("     done after {: 1.2f} sec".format(end_time - start_time))

    return trial_data


# ---------------
# %% cross-validate response timing

# This function was only really working because of a feature in high_trial_count data. 
# It was missampling the trials in standard. I fixed this to index both correctly. 
# Going to leave a legacy version in the code but moving to this new on. 
# Also because the pipeline is not doing a great job of filtering sig cells right now I added 
# lines to redo that in the function.


# def xval_trial_data_OLD(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', signif_only=True, which_neurons='non_stimd', trial_data=None, rng=None):

#     """
#     xval_trial_data(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', signif_only=True, which_neurons='non_stimd', trial_data=None, rng=None)
#     cross-validates response timing for a given area and experiment type
    
#     INPUTS:
#         area         : str, 'V1' or 'M2' (default is 'M2')
#         params       : dict, analysis parameters (default is params from top of this script)
#         expt_type    : str, 'standard', 'short_stim', 'high_trial_count' (default), 'multi_cell'
#         resp_type    : str, 'dff' (default) or 'deconv'
#         signif_only  : bool, if True only include significant neurons (default is True)
#         which_neurons: str, 'non_stimd' (default), 'all', 'stimd'
#         trial_data   : dict with trial data, output of get_single_trial_data (optional, if not provided will call that method)
#         rng          : numpy random number generator (optional)
        
#     OUTPUTS:
#         timing_stats : dict with summary stats and analysis results
#     """
#     # set random seed and delete low /  high tau values if applicable
#     start_time      = time.time()
#     print('Cross-validating reponse timing...')
#     if rng is None:
#         rng = np.random.default_rng(seed=params['random_seed'])
        
#     # get data if necessary
#     if trial_data is None:    
#         trial_data = get_single_trial_data(area=area, params=params, expt_type=expt_type, resp_type=resp_type, signif_only=signif_only, which_neurons=which_neurons, relax_timing_criteria=params['xval_relax_timing_criteria'])
        
#     # loop through rois and stims    
#     sem_overall  = list()
#     median_half1 = list()
#     median_half2 = list()
#     unique_rois  = list(np.unique(trial_data['roi_ids']))
    
#     for roi in unique_rois:
#         ridx         = trial_data['roi_ids']==roi
#         these_trials = trial_data['trial_ids'][ridx]
#         these_stims  = trial_data['stim_ids'][ridx]
#         coms         = trial_data['com_sec'][ridx]
#         peaks        = trial_data['peak_or_trough_time_sec'][ridx]
#         trial_dffs   = list(np.array(trial_data['trig_dff_trials'])[ridx])
#         t_axes       = list(np.array(trial_data['time_axis_sec'])[ridx])
#         unique_stims = list(np.unique(these_stims))
        
#         # take random halves of trials and compare timing stats for each 
#         for stim in unique_stims:
#             tidx        = these_trials[these_stims==stim]-1
#             tidx_shuff  = deepcopy(tidx)
#             ntrials     = np.size(tidx)
#             timing_set1 = np.zeros(params['xval_num_iter'])
#             timing_set2 = np.zeros(params['xval_num_iter'])
#             taxis       = t_axes[tidx[0]]
#             frame_int   = np.diff(taxis)[0]
            
#             # randomly permute trials 
#             for iShuff in range(params['xval_num_iter']):
#                 rng.shuffle(tidx_shuff)
#                 # half1 = tidx_shuff[:np.floor(ntrials/2).astype(int)]
#                 # half2 = tidx_shuff[np.floor(ntrials/2).astype(int)+1:]
#                 n_half = ntrials // 2
#                 half1 = tidx_shuff[:n_half]
#                 half2 = tidx_shuff[n_half:]
                
#                 # get averages of peak (com), or compute avgs for each trial split and peak from there
#                 if params['xval_recompute_timing']:
#                     # trial avgs
#                     trial_avg1 = np.zeros(np.size(t_axes[tidx[0]]))
#                     for idx in list(half1):
#                         trial_avg1 += trial_dffs[idx]
#                     trial_avg1 = trial_avg1 / len(half1)
                    
#                     trial_avg2 = np.zeros(np.size(t_axes[tidx[0]]))
#                     for idx in list(half2):
#                         trial_avg2 += trial_dffs[idx]
#                     trial_avg2 = trial_avg2 / len(half2)
                    
#                     # smooth for peak (com), extract that
#                     smoothed1 = general_stats.moving_average(trial_avg1,num_points=np.round(0.2/frame_int).astype(int)).flatten()
#                     if np.sum(np.isnan(smoothed1)) > 0:
#                         post_idx = int(np.argwhere(np.isnan(smoothed1))[-1]+1)
#                     else:
#                         post_idx = int(np.argwhere(taxis>0)[0])
#                     com1, peak1, trough1 = twop_opto_analysis.response_time_stats(smoothed1[post_idx:],taxis[post_idx:])
                    
#                     smoothed2 = general_stats.moving_average(trial_avg2,num_points=np.round(0.2/frame_int).astype(int)).flatten()
#                     com2, peak2, trough2 = twop_opto_analysis.response_time_stats(smoothed2[post_idx:],taxis[post_idx:])
                    
#                     # collect relevant stat
#                     if params['xval_timing_metric'] == 'peak':
#                         # take peak or trough, whichever is larger
#                         if np.max(smoothed1+smoothed2) > np.abs(np.min(smoothed1+smoothed2)):
#                             timing_set1[iShuff] = peak1
#                             timing_set2[iShuff] = peak2
#                             if iShuff == 0:
#                                 sem_overall.append(np.std([peak1,peak2])/np.sqrt(len(tidx)-1))
#                         else:
#                             timing_set1[iShuff] = trough1
#                             timing_set2[iShuff] = trough2
#                             if iShuff == 0:
#                                 sem_overall.append(np.std([trough1,trough2])/np.sqrt(len(tidx)-1))
                        
#                     elif params['xval_timing_metric'] == 'com':
#                         timing_set1[iShuff] = com1
#                         timing_set2[iShuff] = com2
#                         if iShuff == 0:
#                             sem_overall.append(np.std([com1,com2])/np.sqrt(len(tidx)-1))
                        
#                     else:
#                         print('unknown parameter value for timing metric, returning nothing')
#                         return None
                
#                 # or just take a median of pre-computed trial-by-trial features    
#                 else:
#                     if params['xval_timing_metric'] == 'peak':
#                         timing_set1[iShuff] = np.nanmedian(peaks[half1])
#                         timing_set2[iShuff] = np.nanmedian(peaks[half2])
#                         if iShuff == 0:
#                             sem_overall.append(np.std(peaks)/np.sqrt(len(tidx)-1))
                        
#                     elif params['xval_timing_metric'] == 'com':
#                         timing_set1[iShuff] = np.nanmedian(coms[half1])
#                         timing_set2[iShuff] = np.nanmedian(coms[half2])
#                         if iShuff == 0:
#                             sem_overall.append(np.std(coms)/np.sqrt(len(tidx)-1))
                        
#                     else:
#                         print('unknown parameter value for timing metric, returning nothing')
#                         return None
            
#             # collect median metric for each half
#             median_half1.append(np.nanmedian(timing_set1))
#             median_half2.append(np.nanmedian(timing_set2))
            
#     end_time = time.time()
#     print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
#     xval_results = {
#                 'timing_metric'     : params['xval_timing_metric'], 
#                 'trial_sem'         : np.array(sem_overall).flatten(), 
#                 'median_trialset1'  : np.array(median_half1).flatten(),
#                 'median_trialset2'  : np.array(median_half2).flatten(), 
#                 'response_type'     : resp_type, 
#                 'experiment_type'   : expt_type, 
#                 'which_neurons'     : which_neurons,
#                 'analysis_params'   : deepcopy(params)
#                 }
    
    
#     return xval_results, trial_data
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
        
    sampling_interval=0.032958316
    
    rng = np.random.default_rng(seed=params['random_seed'])
    sem_overall  = list()
    median_half1 = list()
    median_half2 = list()
    unique_rois  = list(np.unique(trial_data['roi_ids']))
    
    for roi in unique_rois:
        ridx = trial_data['roi_ids'] == roi
        
        # Load raw values
        these_trials = np.array(trial_data['trial_ids'])[ridx]
        these_stims  = np.array(trial_data['stim_ids'])[ridx]
        coms         = np.array(trial_data['com_sec'])[ridx]
        peaks        = np.array(trial_data['peak_or_trough_time_sec'])[ridx]
        trial_dffs   = list(np.array(trial_data['trig_dff_trials'])[ridx])
        t_axes       = np.array(trial_data['time_axis_sec'], dtype=object)[ridx]
        unique_stims = list(np.unique(these_stims))
    
        # take random halves of trials and compare timing stats for each 
        for stim in unique_stims:
            tidx        = these_trials[these_stims==stim]-1
            tidx = np.array(tidx, dtype=int)
            tidx_shuff  = deepcopy(tidx)
            ntrials     = np.size(tidx)
            timing_set1 = np.zeros(params['xval_num_iter'])
            timing_set2 = np.zeros(params['xval_num_iter'])
            # timing_set1 = np.zeros(5)
            # timing_set2 = np.zeros(5)
            taxis       = t_axes[tidx[0]]
            frame_int   = np.diff(taxis)[0]
            
            
            peak_array_m2, auc_array_m2, fhwm_m2, peak_times, com_values, com_times = process_trig_dff_trials(
                [trial_dffs[i] for i in tidx],
                kernel_size=15,
                peak_window=[0, 250],
                use_prominence=False,
                prominence_val=1.96
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
                # print(f"Skipping stim {stim}: only {valid_frac*100:.1f}% valid trials")
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

def plot_trial_xval(area='M2', params=params, expt_type='high_trial_count', resp_type='dff',
                    signif_only=True, which_neurons='non_stimd', xval_results=None,
                    rng=None, axis_handle=None, fig_handle=None, trial_data=None,
                    xlim=None, ylim=None, clim=None):
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
        ADDITIONAL OPTIONAL INPUTS:
        xlim : tuple or list (min, max) for x-axis limits
        ylim : tuple or list (min, max) for y-axis limits
        clim : tuple or list (min, max) for color/SEM limits
        
    OUTPUTS:
        ax           : axis handle
        fig          : figure handle
        xval_results : dict with xval results
    """
    
    # run analysis if necessary
    if xval_results is None:
        xval_results, _ = xval_trial_data(area=area, params=params, expt_type=expt_type,
                                          resp_type=resp_type, signif_only=signif_only,
                                          which_neurons=which_neurons, rng=rng,
                                          trial_data=trial_data)

    # plot
    if axis_handle is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle
        fig = fig_handle

    half1 = xval_results['median_trialset1']
    half2 = xval_results['median_trialset2']
    sem = xval_results['trial_sem']

    # set default axis limits if not provided
    if xlim is None or ylim is None:
        xy_all = np.concatenate((half1, half2))
        xy_lim = [np.min(xy_all) - .5, np.max(xy_all) + .5]
        if xlim is None: xlim = xy_lim
        if ylim is None: ylim = xy_lim

    # get colormap
    this_cmap = plt.colormaps.get_cmap('copper')
    this_cmap.set_bad(color='w')

    # scatter plot
    sc = ax.scatter(x=half1, y=half2, c=sem, cmap=this_cmap, edgecolors=[.5, .5, .5], linewidths=.5)

    # axis settings
    ax.plot(xlim, ylim, '--', color=[.8, .8, .8])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if 'peak' in xval_results['timing_metric']:
        ax.set_xlabel('Median peak time, half 1 (sec)')
        ax.set_ylabel('Median peak time, half 2 (sec)')
    else:
        ax.set_xlabel('Median COM time, half 1 (sec)')
        ax.set_ylabel('Median COM time, half 2 (sec)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # colorbar
    cb = fig.colorbar(sc, label='S.E.M. (sec)')
    if clim is not None:
        sc.set_clim(clim)
        # cb.set_clim(clim)

    return ax, fig, xval_results
        
# ---------------
# %% plot average response heatmap
def plot_avg_response_heatmap(area, params=params, expt_type='standard', resp_type='dff',
                              signif_only=True, which_neurons='non_stimd', avg_data=None,
                              axis_handle=None, fig_handle=None, norm_type='minmax',
                              fig_size=(4,6), colorbar_shrink=0.5):
    """
    plot_avg_response_heatmap(area, ...)
    Plots average response heatmap for an area for each ROI, sorted by peak time.

    INPUTS:
        area           : str, 'V1' or 'M2'
        params         : dict, parameters dictionary
        expt_type      : str, 'standard', 'high_trial_count', 'short_stim', 'multi_cell'
        colorbar_shrink: float, scale of colorbar size relative to axis (default=0.5)
    """
    if avg_data is None:
        avg_data = get_avg_trig_responses(area, params=params, expt_type=expt_type,
                                          resp_type=resp_type, signif_only=signif_only,
                                          which_neurons=which_neurons)
        
    # sort by peak time
    idx         = np.argsort(avg_data['peak_times_sec']).flatten()
    resp_mat    = deepcopy(avg_data['trig_dff_avgs'])[idx,:]
    num_neurons = np.size(resp_mat, axis=0)
    
    # normalize
    if norm_type == 'minmax':
        for iNeuron in range(num_neurons):
            resp_mat[iNeuron, :] -= np.nanmin(resp_mat[iNeuron, :])
            resp_mat[iNeuron, :] /= np.nanmax(resp_mat[iNeuron, :])
        cm_name = 'bone'
        lbl     = 'Normalized response'
    elif norm_type == 'absmax':
        for iNeuron in range(num_neurons):
            resp_mat[iNeuron, :] /= np.nanmax(np.abs(resp_mat[iNeuron, :]))
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
    xticks = range(-2, np.ceil(t_axis[-1]).astype(int), 2)
    xpos   = [np.argwhere(t_axis >= x).flatten()[0] for x in xticks]

    # plot
    if axis_handle is None:
        fig = plt.figure(figsize=fig_size)
        ax  = plt.gca()
    else:
        ax  = axis_handle
        fig = fig_handle

    im = ax.imshow(resp_mat, cmap=cm_name, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, label=lbl, shrink=colorbar_shrink)

    ax.set_xticks(np.array(xpos).astype(int))
    ax.set_xticklabels(xticks)
    ax.set_xlabel('Time from stim (sec)')
    ax.set_ylabel('Significant responses (neurons*stim)')
    ax.set_title(params['general_params']['{}_lbl'.format(area)])

    return ax, fig, avg_data

# ---------------
# %% plot grand average of response time course
def plot_response_grand_average(params=params, expt_type='standard', resp_type='dff', signif_only=True, which_neurons='non_stimd', v1_data=None, m2_data=None, axis_handle=None, norm_type='peak',normalize_avg=True,fig_size=(4,4)):

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
        
    # Compute average and SEM without normalization
    v1_avg = np.nanmean(resp_mat_v1, axis=0)
    v1_sem = np.nanstd(resp_mat_v1, axis=0) / np.sqrt(num_v1 - 1)
    m2_avg = np.nanmean(resp_mat_m2, axis=0)
    m2_sem = np.nanstd(resp_mat_m2, axis=0) / np.sqrt(num_m2 - 1)
    
    # Normalize if flag is set
    if normalize_avg:
        # V1 normalization
        v1_min = np.nanmin(v1_avg)
        v1_max = np.nanmax(v1_avg)
        v1_range = v1_max - v1_min
        if v1_range != 0:
            v1_avg = (v1_avg - v1_min) / v1_range
            v1_sem = v1_sem / v1_range
    
        # M2 normalization
        m2_min = np.nanmin(m2_avg)
        m2_max = np.nanmax(m2_avg)
        m2_range = m2_max - m2_min
        if m2_range != 0:
            m2_avg = (m2_avg - m2_min) / m2_range
            m2_sem = m2_sem / m2_range
    

    
    
    # plot
    if axis_handle is None:
        plt.figure(figsize=fig_size)
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
                
                
        # is_good_tau = np.where(np.array(taus) > -0.2, 1, is_good_tau)
        # is_good_tau = np.where(np.array(taus) > -0.2, 1, is_good_tau)
        is_good_tau = np.where(np.array(taus) < 0.1, 0, is_good_tau)


        
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
    
    
    # This is where Probability calc start
    # get responding stats by stimd and non-stimd tau, overall and across time
    unique_sess = list(np.unique(sess_ids))
    ct_short    = 0
    ct_long     = 0
    tau_mat_counts       = np.zeros((2,2))
    tau_mat              = np.zeros((2,2))
    tau_t_bins           = params['tau_by_time_bins']
    
    ### NO THESE ARE ALL INITIALIZED AS shared References!! Avoid in future will lead to each array being exactly 
    # the same even after setting new values
    
    # tau_mat_t_counts     = [np.zeros((2,2))]*(len(tau_t_bins)-1)
    # tau_mat_t            = [np.zeros((2,2))]*(len(tau_t_bins)-1) # normed by cells in each bin
    # tau_mat_t_by_overall = [np.zeros((2,2))]*(len(tau_t_bins)-1) # normed by overall responding cells
    # t_counts             = [np.zeros(2)]*(len(tau_t_bins)-1)   

    tau_mat_t_counts     = [np.zeros((2,2)) for _ in range(len(tau_t_bins)-1)]
    tau_mat_t            = [np.zeros((2,2)) for _ in range(len(tau_t_bins)-1)] # normed by cells in each bin
    tau_mat_t_by_overall = [np.zeros((2,2)) for _ in range(len(tau_t_bins)-1)] # normed by overall responding cells
    t_counts = [np.zeros(2) for _ in range(len(tau_t_bins)-1)]
    
   
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
        
        
        long_tau_nonstimd_denominator  = np.logical_and(np.logical_and(tau > tau_th, is_stimd==0),is_good_tau==1)
        short_tau_nonstimd_denominator = np.logical_and(np.logical_and(tau <= tau_th, is_stimd==0),is_good_tau==1)
        
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
def plot_opto_vs_tau_comparison(area=None, plot_what='prob', params=params, expt_type='standard', resp_type='dff', dff_type='residuals_dff', tau_vs_opto_comp_summary=None, analysis_results_v1=None, analysis_results_m2=None, axis_handles=None, fig_handle=None):
    
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
                fig = fig_handle



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
                    
# ---------------
# %% plot taus on FOV
def plot_resp_fov(area, which_sess=0, which_stim=0, expt_type='standard', resp_type='dff', plot_what='peak_mag', prctile_cap=[0,98], signif_only=False, highlight_signif=True, axis_handle=None,response_stats=None,max_min=None):

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
        if response_stats is None:
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
    
    cell_keys=np.array(response_stats['cell_keys'])[stim_idx].tolist()
    
    keys_to_remove = ['trigdff_param_set_id', 'stim_id']

    # Create a new list with those keys removed
    
    def clean_key(d, keys_to_remove):
        return {k: v for k, v in d.items() if k not in keys_to_remove}
    
    cleaned_cell_keys = [clean_key(d, keys_to_remove) for d in cell_keys]
    # _ = None  # suppress unwanted output
    
    stim_key=cleaned_cell_keys[-1]
    
# This was a fucking nightmarem the issue comes from a few things but mainly how fetch resorts after a fetch by roi_id. in fullresp stat the way we collect the dat for 'all'
#  is by appending the stim index on the responsive index then this creates confusion about keys and position indexing. need to fix long term, came up with some solution but
#  that has its own issues, for now this fix actually plots right cells

    roi_keys   = np.array(response_stats['roi_keys'])[stim_idx].tolist()
    roi_keys = [d for d in roi_keys if d != stim_key]
    
    # Fetch ROI coordinates
    row_pxls, col_pxls = (VM['twophoton'].Roi2P & roi_keys).fetch('row_pxls', 'col_pxls')
    
    # Fetch stim coordinates (assumed to be single-element arrays)
    row_pxls_stim, col_pxls_stim = (VM['twophoton'].Roi2P & stim_key).fetch('row_pxls', 'col_pxls')
    
    # Prepend the stim coordinate
    row_pxls = list(row_pxls)+list(row_pxls_stim) 
    col_pxls = list(col_pxls)+list(col_pxls_stim) 
        
    roi_coords = [row_pxls,col_pxls]
    
    num_rows,num_cols,um_per_pxl = (VM['twophoton'].Scan & roi_keys[0]).fetch1('lines_per_frame','pixels_per_line','microns_per_pxl_y')
    im_size    = (num_rows,num_cols)
    
    
    # stimd_idx  = np.argwhere(response_stats['is_stimd'][stim_idx]).flatten().tolist()
    stimd_idx  = [cell_keys[-1]['roi_id']]
    # stimd_idx  = [roi_keys[-1]['roi_id']]
    
    
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
                
            iax, fig = plot_fov_heatmap_blur(roi_vals=fvals[:,iF].tolist(), roi_coords=roi_coords, im_size=im_size, um_per_pxl=um_per_pxl, \
                                        prctile_cap=prctile_cap, cbar_lbl=lbl, axisHandle=ax[iF], figHandle=fig, \
                                        cmap='coolwarm', background_cl = 'gray', plot_colorbar=cbar, max_min=[-imax,imax])
            
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
                    iax.text(x,y,'*',color='w',fontsize=8)
                
            ax[iF] = iax
    else:
        if plot_what == 'peak_mag':
            imax      = np.percentile(np.abs(vals),prctile_cap[1])
            imin      = -imax
            maxmin    = [imin,imax]
            cmap_name = 'coolwarm'
        else:
            maxmin    = None
            cmap_name = 'Blues'
        
        if max_min is not None:
            maxmin     = max_min
            
            
        ax, fig = plot_fov_heatmap_blur(roi_vals=vals.tolist(), roi_coords=roi_coords, im_size=im_size, um_per_pxl=um_per_pxl, \
                                    prctile_cap=prctile_cap, cbar_lbl=lbl, axisHandle=ax, figHandle=fig, \
                                    cmap=cmap_name, background_cl = 'gray',max_min=maxmin)

        # add arrow on stim'd neuron
        for istim in stimd_idx:
            x = np.median(roi_coords[1][-1])
            y = np.median(roi_coords[0][-1]) - 15
            ax.plot(x,y,'kv',ms=6)
        
        # draw a circle around significant neurons 
        
        if highlight_signif:
            for isig in is_sig.tolist():
                if isig in stimd_idx:
                    continue  # Skip stimulated ROIs
                else:
                # Compute asterisk position
                    x = np.median(roi_coords[1][isig]) + 8
                    y = np.median(roi_coords[0][isig]) + 14
            
                    # Debugging: Print coordinates to check values
                    # print(f"Adding asterisk at ({x}, {y}) for ROI index {isig}")
            
                    # Add text to plot
                    ax.text(x, y, '*', color='k', fontsize=20)
        # if highlight_signif:
        #     for isig in is_sig.tolist():
        #         if isig in stimd_idx:
        #             continue
        #         x = np.median(roi_coords[1][isig]) + 8
        #         y = np.median(roi_coords[0][isig]) + 14
        #         iax.text(x,y,'*',color='w',fontsize=8)
        # If using an interactive plot, refresh it
        plt.draw()
        
    if fig is None:
        fig = axis_handle.get_figure() 
        
    return ax, fig,roi_keys

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
    
    # NC wrote this to exclude session that had no included cells
    valid_expt_keys = []
    for expt_key in expt_keys:
        sess_key = {'subject_fullname': expt_key['subject_fullname'],
                    'session_date'    : expt_key['session_date'],
                    'trigdff_param_set_id': params['trigdff_param_set_id_{}'.format('dff')],
                    'trigdff_inclusion_param_set_id': params['trigdff_inclusion_param_set_id'],
                    }
        incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & sess_key).fetch('is_included', 'KEY')
        if incl.size > 0:
            valid_expt_keys.append(expt_key)
            
    expt_keys=valid_expt_keys
    
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
# %%

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def plot_pca_dist_scatter_dual(
    area,
    params,
    expt_type='standard+high_trial_count',
    resp_type='dff',
    eg_ids=None,
    trial_pca_results_1=None,
    trial_pca_results_2=None,
    label_1='Dataset 1',
    label_2='Dataset 2',
    color_1='blue',
    color_2='green',
    axis_handle=None
):
    """
    Plots baseline vs. post-stim PCA distance scatter for up to two datasets.

    Inputs:
        area               : str, brain area ('V1' or 'M2')
        params             : dict, analysis parameters
        expt_type          : str, experiment type for PCA calculation if not provided
        resp_type          : str, response type ('dff' or 'deconv')
        eg_ids             : list of int, experiment group IDs to restrict to (optional)
        trial_pca_results_1: dict, PCA results for first dataset (optional)
        trial_pca_results_2: dict, PCA results for second dataset (optional)
        label_1            : str, label for dataset 1
        label_2            : str, label for dataset 2
        color_1            : str, color for dataset 1
        color_2            : str, color for dataset 2
        axis_handle        : matplotlib axis (optional)

    Returns:
        ax                 : matplotlib axis
        trial_pca_results_1: dict of PCA results (if computed inside)
    """

    # Run analysis if necessary
    if trial_pca_results_1 is None:
        trial_pca_results_1 = batch_trial_pca(
            area=area,
            params=params,
            expt_type=expt_type,
            resp_type=resp_type,
            eg_ids=eg_ids
        )

    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle

    # Helper function for plotting a single dataset
    def plot_dataset(trial_pca_results, color, label):
        ax.plot(
            trial_pca_results['basel_dists'],
            trial_pca_results['resp_dists'],
            'o', color=color, mew=0, label=label
        )
        olsfit = sm.OLS(trial_pca_results['resp_dists'],
                        sm.add_constant(trial_pca_results['basel_dists'])).fit()
        x_vals = np.linspace(min(trial_pca_results['basel_dists']),
                             max(trial_pca_results['basel_dists']), 100)
        y_vals = olsfit.predict(sm.add_constant(x_vals))
        predci = olsfit.get_prediction(sm.add_constant(x_vals)).summary_frame()
        ci_up = predci['mean_ci_upper']
        ci_low = predci['mean_ci_lower']

        ax.plot(x_vals, y_vals, '-', color=color)
        ax.plot(x_vals, ci_up, '--', color=color, lw=.4)
        ax.plot(x_vals, ci_low, '--', color=color, lw=.4)

        # Show correlation on first dataset only
        if label == label_1:
            yl = ax.get_ylim()
            xl = ax.get_xlim()
            ax.text(xl[0] * 1.05, yl[1] * .95, 'r = {:1.2f}'.format(trial_pca_results['corr_basel_vs_resp']))
            ax.text(xl[0] * 1.05, yl[1] * .90, 'p = {:1.2g}'.format(trial_pca_results['p_basel_vs_resp']))

    # Plot both datasets
    plot_dataset(trial_pca_results_1, color_1, label_1)

    if trial_pca_results_2 is not None:
        plot_dataset(trial_pca_results_2, color_2, label_2)

    ax.set_xlabel('Trial Euclidean dist. (baseline)')
    ax.set_ylabel('Trial Euclidean dist. (post-stim)')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax, trial_pca_results_1


# %%
import numpy as np
from scipy.signal import medfilt, find_peaks, peak_widths
from scipy.stats import zscore

def process_trig_dff_trials(a, kernel_size=5, z_score=False,
                             peak_window=[0, 250], use_prominence=False,
                             prominence_val=0.1, return_all_peaks=False,
                             peak_thresh=None,
                             trace_start=100,
                             min_width_val=2):  # <- new optional input
    peak_values = []
    auc_values = []
    fwhm_values = []
    peak_times = []
    com_values = []
    com_times = []

    for i, trace in enumerate(a):
        # Skip if trace is invalid
        if trace is None or len(trace) < 121 or np.all(np.isnan(trace)):
            peak_values.append(np.nan)
            auc_values.append(np.nan)
            fwhm_values.append(np.nan)
            peak_times.append(np.nan)
            com_values.append(np.nan)
            com_times.append(np.nan)
            continue

        # Optional z-scoring
        if z_score:
            trace = zscore(trace, nan_policy='omit')

        # Apply median filter to the trace starting at index 100
        filtered = medfilt(trace[trace_start:], kernel_size=kernel_size)

        # Normalize to have min = 0
        filtered = filtered - np.nanmin(filtered)

        # AUC of the filtered trace
        auc = np.nansum(filtered)

        # Peak detection (within window)
        pw_start, pw_end = peak_window
        signal_segment = filtered[pw_start:pw_end]
        
        # Prepend a 0 to help detect early peaks
        signal_segment_padded = np.insert(signal_segment, 0, 0)
        

        # Set find_peaks parameters
        peak_kwargs = {}
        if use_prominence:
            peak_kwargs['prominence'] = prominence_val
        peak_kwargs['width'] = min_width_val  # Set your minimum width here
        
        peaks, properties = find_peaks(signal_segment, **peak_kwargs)

        # # Adjust peaks indices because of the prepended 0
        # peaks = peaks - 1
        
        # # Remove any peaks that are now at negative index due to shift
        # valid_indices = peaks >= 0
        # peaks = peaks[valid_indices]
        # for key in properties:
        #     properties[key] = properties[key][valid_indices]

        if len(peaks) == 0:
            peak_val = np.nan
            fwhm = np.nan
            main_peak = np.nan
        else:
            # Select the most prominent or highest peak
            if return_all_peaks:
                peak_val = np.nanmean(signal_segment[peaks])
                widths_result = peak_widths(signal_segment, peaks, rel_height=0.5)
                fwhm = np.nanmean(widths_result[0])
            else:
                if use_prominence:
                    main_peak_idx = np.argmax(properties['prominences'])
                else:
                    main_peak_idx = np.argmax(signal_segment[peaks])

                main_peak = peaks[main_peak_idx]
                peak_val = signal_segment[main_peak]

                # Calculate FWHM using peak_widths
                widths_result = peak_widths(signal_segment, [main_peak], rel_height=0.5)
                fwhm = widths_result[0][0]

        # Center of mass within the peak window
        window_vals = signal_segment
        time_indices = np.arange(pw_start, pw_end)
        window_vals_sum = np.nansum(window_vals)

        if window_vals_sum == 0 or np.isnan(window_vals_sum):
            com = np.nan
            com_time = np.nan
        else:
            com = np.nansum(time_indices * window_vals) / window_vals_sum
            com_time = int(np.round(com))

        # Apply threshold to nan out values if peak is below threshold
        if peak_thresh is not None and (np.isnan(peak_val) or peak_val < peak_thresh):
            peak_val = np.nan
            auc = np.nan
            fwhm = np.nan
            main_peak = np.nan
            com = np.nan
            com_time = np.nan

        # Append metrics
        peak_values.append(peak_val)
        auc_values.append(auc)
        fwhm_values.append(fwhm)
        peak_times.append(main_peak)
        com_values.append(com)
        com_times.append(com_time)

    return (np.array(peak_values),
            np.array(auc_values),
            np.array(fwhm_values),
            np.array(peak_times),
            np.array(com_values),
            np.array(com_times))
# %%
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr

def trial_peak_reliability(trial_dffs, idx_start, idx_end, min_trials=5):
    """
    Compute average pairwise Pearson correlation across trials
    in a given window.

    Args:
        trial_dffs: list of 1D numpy arrays (trial-wise traces)
        idx_start: start index of time window
        idx_end: end index of time window (exclusive)
        min_trials: minimum trials to consider

    Returns:
        mean_corr: mean pairwise Pearson correlation
    """
    windowed = [trace[idx_start:idx_end] for trace in trial_dffs]
    windowed = [w for w in windowed if not np.any(np.isnan(w))]

    n_trials = len(windowed)
    if n_trials < min_trials:
        return np.nan

    corrs = []
    for i, j in combinations(range(n_trials), 2):
        r, _ = pearsonr(windowed[i], windowed[j])
        corrs.append(r)

    return np.nanmean(corrs) if corrs else np.nan

# %%
import numpy as np

def create_pseudo_trials_from_baseline(
    baseline_dff,
    number_of_trials,
    trial_length,
    baseline_start_idx=0,
    max_offset=None,
    zscore_to_baseline=True,
    within_trial_baseline_frames=80,
    random_seed=None,
    bootstrap=True,
    min_spacing_between_starts=50
):
    """
    Create pseudo-trials from a long baseline dF/F trace.
    Each pseudo-trial includes a baseline + response window. If z-scoring is enabled,
    the first N frames of each trial are used as the within-trial baseline.

    Parameters
    ----------
    baseline_dff : 1D array
        Long baseline trace (e.g., from spontaneous period).
    number_of_trials : int
        Number of pseudo-trials to generate.
    trial_length : int
        Length of each pseudo-trial (including baseline + response).
    baseline_start_idx : int
        Index where slicing trials is allowed to begin.
    max_offset : int or None
        Optional maximum index for trial start (defaults to end of array - trial_length).
    zscore_to_baseline : bool
        Whether to z-score using the first N frames of each trial.
    within_trial_baseline_frames : int
        Number of frames at the start of each trial to use as the baseline for z-scoring.
    random_seed : int or None
        Seed for reproducibility.
    bootstrap : bool
        Sample with replacement (True) or not (False).
    min_spacing_between_starts : int
        Minimum distance (in frames) between trial start points (only applies if bootstrap=False).

    Returns
    -------
    pseudo_trials : list of np.ndarray
        List of pseudo-trials (each length = trial_length).
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    baseline_dff = np.array(baseline_dff).flatten()
    total_frames = len(baseline_dff)

    max_start_idx = total_frames - trial_length
    if max_offset is not None:
        max_start_idx = min(max_start_idx, max_offset)

    possible_starts = np.arange(baseline_start_idx, max_start_idx)

    if bootstrap or min_spacing_between_starts == 0:
        if len(possible_starts) < number_of_trials and not bootstrap:
            raise ValueError("Not enough baseline data to sample requested number of trials.")
        start_idxs = np.random.choice(possible_starts, size=number_of_trials, replace=bootstrap)
    else:
        # Enforce spacing constraint between start indices
        start_idxs = []
        possible_starts_set = set(possible_starts)

        while len(start_idxs) < number_of_trials:
            if not possible_starts_set:
                raise ValueError("Cannot find enough non-overlapping start indices with given spacing.")
            new_start = np.random.choice(list(possible_starts_set))
            start_idxs.append(new_start)

            # Remove indices within min_spacing from the selected start
            exclusion_zone = set(range(new_start - min_spacing_between_starts + 1,
                                       new_start + min_spacing_between_starts))
            possible_starts_set.difference_update(exclusion_zone)

    pseudo_trials = []

    for start_idx in start_idxs:
        trial = baseline_dff[start_idx : start_idx + trial_length].copy()

        if zscore_to_baseline:
            if within_trial_baseline_frames > len(trial):
                raise ValueError("Baseline window for z-scoring exceeds trial length.")
            baseline = trial[:within_trial_baseline_frames]
            baseline_mean = np.nanmean(baseline)
            baseline_std = np.nanstd(baseline)
            if baseline_std == 0 or np.isnan(baseline_std):
                trial_z = np.zeros_like(trial)
            else:
                trial_z = (trial - baseline_mean) / baseline_std
            pseudo_trials.append(trial_z)
        else:
            pseudo_trials.append(trial)

    return pseudo_trials


# %%

def create_pseudo_trials_from_baseline_non_overlaping(
    baseline_dff,
    number_of_trials,
    trial_length,
    baseline_start_idx=0,
    max_offset=None,
    zscore_to_baseline=True,
    within_trial_baseline_frames=50,
    random_seed=None
):
    """
    Create pseudo-trials from a long baseline dF/F trace without overlap.

    Parameters:
    - baseline_dff: 1D array of baseline activity
    - number_of_trials: how many pseudo-trials to create
    - trial_length: number of frames per trial
    - baseline_start_idx: where the baseline region starts
    - max_offset: optional, max index allowed for slicing trials
    - zscore_to_baseline: whether to z-score each trial to a pre-trial baseline
    - baseline_window_for_z: number of frames before each trial used for z-scoring
    - random_seed: optional for reproducibility

    Returns:
    - List of pseudo-trial arrays (length = number_of_trials)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    baseline_dff = np.array(baseline_dff).flatten()
    max_start = len(baseline_dff) - trial_length
    if max_offset is not None:
        max_start = min(max_start, max_offset)

    # Generate non-overlapping valid start indices
    valid_starts = np.arange(baseline_start_idx, max_start, trial_length)
    if len(valid_starts) < number_of_trials:
        raise ValueError("Not enough non-overlapping baseline segments for the requested number of trials.")

    start_idxs = np.random.choice(valid_starts, size=number_of_trials, replace=False)

    pseudo_trials = []
    for start_idx in start_idxs:
        # Extract full trial window
        trial = baseline_dff[start_idx : start_idx + trial_length].copy()
    
        if zscore_to_baseline:
            if within_trial_baseline_frames > len(trial):
                raise ValueError("Baseline window for z-scoring exceeds trial length.")
            baseline = trial[:within_trial_baseline_frames]
            baseline_mean = np.nanmean(baseline)
            baseline_std = np.nanstd(baseline)
            if baseline_std == 0 or np.isnan(baseline_std):
                trial_z = np.zeros_like(trial)
            else:
                trial_z = (trial - baseline_mean) / baseline_std
            pseudo_trials.append(trial_z)
        else:
            pseudo_trials.append(trial)
    
    return pseudo_trials


# %%
import pandas as pd
import numpy as np

def calculate_proportion_by_distance_bin(df,
                                         group_cols,
                                         dist_col,
                                         bins=[0, 50, 100, 150, 200, 250, 300, np.inf]):
    """
    Bins cells by distance and calculates the proportion of cells in each bin for each group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a column representing distance from stimulation.
    group_cols : list of str
        Columns to group by (e.g., subject/session/stim identifiers).
    dist_col : str
        Name of the column containing distance values.
    bins : list
        Bin edges for distance (default = [0, 50, ..., 300, np.inf]).

    Returns
    -------
    proportion_df : pd.DataFrame
        Wide-form DataFrame with one row per group and columns for each distance bin's proportion.
    just_values_T : pd.DataFrame
        Transposed DataFrame with rows = distance bins and columns = group combinations.
    bin_labels : list of str
        Labels used for the distance bins (in order).
    """
    # Generate bin labels
    bin_labels = [
        f"{int(bins[i])}-{int(bins[i+1])}" if np.isfinite(bins[i+1]) else f"{int(bins[i])}+"
        for i in range(len(bins) - 1)
    ]

    # Bin the distance values
    df = df.copy()
    df['dist_bin'] = pd.cut(df[dist_col], bins=bins, labels=bin_labels, right=False)

    # Count per group and bin
    count_df = df.groupby(group_cols + ['dist_bin']).size().reset_index(name='count')

    # Total per group
    total_df = df.groupby(group_cols).size().reset_index(name='total')

    # Merge and calculate proportion
    merged = pd.merge(count_df, total_df, on=group_cols)
    merged['proportion'] = merged['count'] / merged['total']

    # Pivot to wide format
    proportion_df = merged.pivot_table(index=group_cols,
                                       columns='dist_bin',
                                       values='proportion',
                                       fill_value=0)

    # Clean up column names
    proportion_df.columns.name = None
    proportion_df = proportion_df.reset_index()

    # Extract just the values and transpose
    just_values = proportion_df.drop(columns=group_cols)
    just_values_T = just_values.transpose()
    
    # Calculate bin centers
    bin_edges = np.array(bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


    return proportion_df, just_values_T, bin_labels, bin_centers

# %%

def calculate_responsive_proportion_by_distance_bin_by_all_cells(df_responsive,
                                                    df_all,
                                                    group_cols,
                                                    dist_col,
                                                    bins=[0, 50, 100, 150, 200, 250, 300, np.inf]):
    """
    Bins cells by distance and calculates the proportion of responsive cells
    relative to all cells in each bin, grouped by specified columns.

    Parameters
    ----------
    df_responsive : pd.DataFrame
        DataFrame containing only responsive cells.
    df_all : pd.DataFrame
        DataFrame containing all cells (including responsive ones).
    group_cols : list of str
        Columns to group by (e.g., subject/session/stim identifiers).
    dist_col : str
        Column containing distance values.
    bins : list
        Bin edges for distance (default = [0, 50, ..., 300, np.inf]).

    Returns
    -------
    proportion_df : pd.DataFrame
        Wide-form DataFrame with one row per group and columns for each distance bin's proportion.
    just_values_T : pd.DataFrame
        Transposed DataFrame with rows = distance bins and columns = group combinations.
    bin_labels : list of str
        Labels used for the distance bins (in order).
    bin_centers : np.ndarray
        Numerical centers of each distance bin.
    """
    bin_labels = [
        f"{int(bins[i])}-{int(bins[i+1])}" if np.isfinite(bins[i+1]) else f"{int(bins[i])}+"
        for i in range(len(bins) - 1)
    ]

    # Bin distances
    df_responsive = df_responsive.copy()
    df_all = df_all.copy()
    df_responsive['dist_bin'] = pd.cut(df_responsive[dist_col], bins=bins, labels=bin_labels, right=False)
    df_all['dist_bin'] = pd.cut(df_all[dist_col], bins=bins, labels=bin_labels, right=False)

    # Count responsive and all cells per group/bin
    responsive_count = df_responsive.groupby(group_cols + ['dist_bin']).size().reset_index(name='responsive_count')
    all_count = df_all.groupby(group_cols + ['dist_bin']).size().reset_index(name='total_count')

    # Merge and calculate proportion
    merged = pd.merge(all_count, responsive_count, on=group_cols + ['dist_bin'], how='left')
    merged['responsive_count'] = merged['responsive_count'].fillna(0)
    merged['proportion'] = merged['responsive_count'] / merged['total_count']

    # Pivot to wide format
    proportion_df = merged.pivot_table(index=group_cols,
                                       columns='dist_bin',
                                       values='proportion',
                                       fill_value=0)

    proportion_df.columns.name = None
    proportion_df = proportion_df.reset_index()

    # Transpose values for plotting or stats
    just_values = proportion_df.drop(columns=group_cols)
    just_values_T = just_values.transpose()

    # Bin centers for plotting
    bin_edges = np.array(bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return proportion_df, just_values_T, bin_labels, bin_centers
# %%
import pandas as pd
import numpy as np

def calculate_condition_proportion_by_bin(df,
                                          group_cols,
                                          bin_col,
                                          condition_col,
                                          bins=[0, 50, 100, 150, 200, 250, 300, np.inf]):
    """
    Bins rows by a numeric column and calculates the proportion of rows where
    condition_col is True in each bin, grouped by group_cols.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : list of str
        Columns to group by (e.g., subject/session/stim identifiers).
    bin_col : str
        Column name to bin by (e.g., distance).
    condition_col : str
        Column name that holds the boolean condition (e.g., True = responsive).
    bins : list
        Bin edges for binning (default = [0, 50, ..., 300, np.inf]).

    Returns
    -------
    proportion_df : pd.DataFrame
        Wide-form DataFrame with one row per group and columns for each bin's proportion.
    just_values_T : pd.DataFrame
        Transposed DataFrame with rows = bins and columns = group combinations.
    bin_labels : list of str
        Labels used for the bins (in order).
    bin_centers : np.ndarray
        Numerical centers of the bins (for plotting).
    """
    # Generate bin labels
    bin_labels = [
        f"{int(bins[i])}-{int(bins[i+1])}" if np.isfinite(bins[i+1]) else f"{int(bins[i])}+"
        for i in range(len(bins) - 1)
    ]

    # Bin the bin_col values
    df = df.copy()
    df['bin'] = pd.cut(df[bin_col], bins=bins, labels=bin_labels, right=False)

    # Count total per group + bin
    total_counts = df.groupby(group_cols + ['bin']).size().reset_index(name='total_count')

    # Count where condition_col is True per group + bin
    condition_counts = df[df[condition_col]].groupby(group_cols + ['bin']).size().reset_index(name='condition_count')

    # Merge counts
    merged = pd.merge(total_counts, condition_counts, on=group_cols + ['bin'], how='left')
    merged['condition_count'] = merged['condition_count'].fillna(0)

    # Calculate proportion
    merged['proportion'] = merged['condition_count'] / merged['total_count']

    # Pivot to wide format
    proportion_df = merged.pivot_table(index=group_cols,
                                       columns='bin',
                                       values='proportion',
                                       fill_value=0)
    proportion_df.columns.name = None
    proportion_df = proportion_df.reset_index()

    # Transpose values
    just_values = proportion_df.drop(columns=group_cols)
    just_values_T = just_values.transpose()

    # Bin centers
    bin_edges = np.array(bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return proportion_df, just_values_T, bin_labels, bin_centers
# %%
import pandas as pd
import numpy as np

def bin_column_values(df, bin_col, value_col, bins):
    """
    Bins one column and collects values from another column into a 2D DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    bin_col : str
        Column to bin (e.g., 'distance').
    value_col : str
        Column with values to collect (e.g., 'dff_peak').
    bins : list
        Bin edges.

    Returns
    -------
    binned_df : pd.DataFrame
        DataFrame where each row is a bin and each column a padded value.
    bin_labels : list
        Labels used for the bins.
    bin_centers : np.ndarray
        Numeric centers of bins.
    """
    df = df.copy()

    # Create bin labels
    bin_labels = [
        f"{int(bins[i])}-{int(bins[i + 1])}" if np.isfinite(bins[i + 1]) else f"{int(bins[i])}+"
        for i in range(len(bins) - 1)
    ]
    
    # Bin the values
    df['bin'] = pd.cut(df[bin_col], bins=bins, labels=bin_labels, right=False)

    # Collect values for each bin
    binned_series = df.groupby('bin')[value_col].apply(list).reindex(bin_labels)

    # Find the max list length
    max_len = binned_series.dropna().apply(len).max()

    # Pad with NaNs and convert to DataFrame
    binned_data = [
        np.pad(vals, (0, max_len - len(vals)), constant_values=np.nan) if isinstance(vals, list) else [np.nan] * max_len
        for vals in binned_series
    ]
    binned_df = pd.DataFrame(binned_data, index=bin_labels, columns=[f"value_{i}" for i in range(max_len)])

    # Bin centers for plotting
    bin_edges = np.array(bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return binned_df, bin_labels, bin_centers


# %%
import pandas as pd

def group_trig_responses_by_expt(roi_keys,stim_ids,trig_dff_avgs):
    """
    Organizes individual ROI average responses into a nested DataFrame grouped by:
    - subject_fullname
    - session_date
    - stim_id

    INPUT:
        summary_data : dict returned by get_avg_trig_responses()

    OUTPUT:
        df_grouped : pd.DataFrame with columns ['subject_fullname', 'session_date', 'stim_id', 'trig_dff_avgs']
                     where 'trig_dff_avgs' is a list of np.arrays for each ROI in that group
    """
    # roi_keys       = summary_data['roi_keys']
    # # expt_keys      = summary_data['roi_keys']['
    # stim_ids       = summary_data['stim_ids']
    # trig_dff_avgs  = summary_data['trig_dff_avgs']

    # Ensure consistent length
    assert len(roi_keys) == len(stim_ids) == len(trig_dff_avgs), "Data mismatch in summary_data."

    # Assemble a flat list of dicts with metadata and average response
    rows = []
    for ek, sid, avg in zip(roi_keys, stim_ids, trig_dff_avgs):
        rows.append({
            'subject_fullname': ek['subject_fullname'],
            'session_date'    : ek['session_date'],
            'stim_id'         : sid,
            'trig_dff_avg'    : avg
        })

    df = pd.DataFrame(rows)

    # Group by subject/session/stim and nest the average responses
    df_grouped = df.groupby(['subject_fullname', 'session_date', 'stim_id'])['trig_dff_avg'].apply(list).reset_index()
    df_grouped.rename(columns={'trig_dff_avg': 'trig_dff_avgs'}, inplace=True)

    return df_grouped
# %%


def align_averaged_traces_from_lists(df, trace_col='averaged_traces_all', time_col='time_axes'):
    """
    Aligns traces to a shared time axis using only the first time axis from each row.

    Parameters:
    - df: DataFrame with columns:
        - trace_col: 1D averaged trace (np.array or list)
        - time_col: list of time arrays per row (use first one only)

    Returns:
    - common_time_df: DataFrame with 1D shared time axis
    - aligned_traces_df: DataFrame with 2D aligned traces
    """
    # Step 1: Extract and trim each trace and time axis to matching lengths
    trace_list = []
    time_list = []

    for _, row in df.iterrows():
        trace = np.array(row[trace_col])
        time = np.array(row[time_col][0])  # use only the first time axis
        n = min(len(trace), len(time))
        trace_list.append(trace[:n])
        time_list.append(time[:n])

    # Step 2: Find common overlapping time range
    start_times = [t[0] for t in time_list]
    end_times = [t[-1] for t in time_list]
    common_start = max(start_times)
    common_end = min(end_times)
    if common_start >= common_end:
        raise ValueError("No overlapping time range across traces.")

    # Step 3: Define common time axis using finest resolution
    resolutions = [np.mean(np.diff(t)) for t in time_list]
    min_res = min(resolutions)
    common_time = np.arange(common_start, common_end + min_res, min_res)

    # Step 4: Interpolate each trace to the common time axis
    interpolated_traces = []
    for trace, time in zip(trace_list, time_list):
        valid_mask = (common_time >= time[0]) & (common_time <= time[-1])
        interp_vals = np.interp(common_time[valid_mask], time, trace)
        interpolated_traces.append(interp_vals)

    # Step 5: Trim all to same length (shortest)
    min_len = min(len(t) for t in interpolated_traces)
    final_traces = np.array([t[:min_len] for t in interpolated_traces])
    final_time = common_time[:min_len]

    # Step 6: Return as DataFrames
    common_time_df = pd.DataFrame(final_time, columns=['time'])
    aligned_traces_df = pd.DataFrame(final_traces)

    return common_time_df, aligned_traces_df
# %%
import numpy as np

def align_traces_from_separate_lists(trials_list, time_axes_list):
    """
    Aligns a list of trials to a shared time axis by interpolating and trimming 
    to the overlapping time range.

    Parameters:
    - trials_list: list of 1D arrays/lists of trace values
    - time_axes_list: list of 1D arrays/lists of time values (same length as trials_list)

    Returns:
    - aligned_traces: list of aligned 1D arrays (all same length)
    - aligned_time_axes: list of 1D arrays, each identical (shared aligned time axis)
    """
    if len(trials_list) != len(time_axes_list):
        raise ValueError("trials_list and time_axes_list must be the same length")

    trace_list = []
    time_list = []

    for trial, time in zip(trials_list, time_axes_list):
        trace = np.array(trial)
        time = np.array(time)
        n = min(len(trace), len(time))
        trace_list.append(trace[:n])
        time_list.append(time[:n])

    # Find common overlapping time range
    start_times = [t[0] for t in time_list]
    end_times = [t[-1] for t in time_list]
    common_start = max(start_times)
    common_end = min(end_times)
    if common_start >= common_end:
        raise ValueError("No overlapping time range across traces.")

    # Define common time axis using finest resolution
    resolutions = [np.mean(np.diff(t)) for t in time_list]
    min_res = min(resolutions)
    common_time = np.arange(common_start, common_end + min_res, min_res)

    # Interpolate each trace to the common time axis (only within each trace's range)
    interpolated_traces = []
    for trace, time in zip(trace_list, time_list):
        valid_mask = (common_time >= time[0]) & (common_time <= time[-1])
        interp_vals = np.interp(common_time[valid_mask], time, trace)
        interpolated_traces.append(interp_vals)

    # Trim all to same minimum length
    min_len = min(len(t) for t in interpolated_traces)
    aligned_traces = [t[:min_len] for t in interpolated_traces]
    aligned_time = common_time[:min_len]

    # Return aligned time axis as a repeated list
    aligned_time_axes = [aligned_time] * len(aligned_traces)

    return aligned_traces, aligned_time_axes
