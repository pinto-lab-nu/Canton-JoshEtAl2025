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
from copy import deepcopy
from schemas import spont_timescales
from schemas import twop_opto_analysis
from utils.stats import general_stats
from utils.plotting import plot_pval_circles
from utils.plotting import plot_fov_heatmap
from analyzeSpont2P import params as tau_params

import time

# connect to dj 
VM     = connect_to_dj.get_virtual_modules()

# %% declare default params, inherit general params from tau_params
params = {
        'random_seed'                    : 42, 
        'trigdff_param_set_id_dff'       : 4, 
        'trigdff_param_set_id_deconv'    : 5, 
        'trigdff_inclusion_param_set_id' : 3,
        'trigdff_inclusion_param_set_id_notiming' : 4, # for inhibition, we may want to relax trough timing constraint
        'trigspeed_param_set_id'         : 1,
        'prop_resp_bins'                 : np.arange(0,.41,.01),
        'dist_bins_resp_prob'            : np.arange(30,480,50),
        'response_magnitude_bins'        : np.arange(-2.5,7.75,.25),
        'response_time_bins'             : np.arange(0,10.1,.1),
        'expression_level_type'          : 'intensity_zscore_stimdcells',
        'xval_relax_timing_criteria'     : True, # set to true will select inclusion criteria that do not enforce peak timing
        'xval_recompute_timing'          : False, # set to true will recompute peak (com) for every xval iteration, false just averages existing peak times
        'xval_timing_metric'             : 'peak', # 'peak' or 'com'. peak is time of peak or trough
        'xval_num_iter'                  : 1000,
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
    
    start_time      = time.time()
    print('Fetching opto-triggered averages...')
    
    # get relevant keys
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    
    # restrict to only desired rec/stim if applicable
    if eg_ids is not None:
        expt_keys = list(np.array(expt_keys)[np.array(eg_ids).astype(int)])
    
    # loop through keys to fetch the responses
    trigdff_param_set_id           = params['trigdff_param_set_id_{}'.format(resp_type)]
    trigdff_inclusion_param_set_id = params['trigdff_inclusion_param_set_id']
    
    avg_resps = list()
    sem_resps = list()
    t_axes    = list()
    is_stimd  = list()
    is_sig    = list()
    coms      = list()
    peak_ts   = list()
    num_expt  = len(expt_keys)
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
            avgs, sems, ts, stimd, com, peak = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('trig_dff_avg', 'trig_dff_sem', 'time_axis_sec', 'is_stimd', 'center_of_mass_sec_overall', 'time_of_peak_sec_overall')
        else:
            sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')
            idx  = np.argwhere(np.array(incl)==1).flatten()
            keys = list(np.array(keys)[idx])
            sig  = list(np.array(sig)[idx])
            
            if signif_only:
                idx  = np.argwhere(np.array(sig)==1).flatten()
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx]) 
            
            if which_neurons == 'all':
                avgs, sems, ts, stimd, com, peak = (twop_opto_analysis.TrigDffTrialAvg & keys).fetch('trig_dff_avg', 'trig_dff_sem', 'time_axis_sec', 'is_stimd', 'center_of_mass_sec_overall', 'time_of_peak_sec_overall')
            elif which_neurons == 'non_stimd':
                avgs, sems, ts, stimd, com, peak = (twop_opto_analysis.TrigDffTrialAvg & keys & 'is_stimd=0').fetch('trig_dff_avg', 'trig_dff_sem', 'time_axis_sec', 'is_stimd', 'center_of_mass_sec_overall', 'time_of_peak_sec_overall')
            else:
                print('Unknown category of which_neurons, returning nothing')
                return None

        # flatten list
        [avg_resps.append(avg) for avg in avgs]
        [sem_resps.append(sem) for sem in sems]
        [t_axes.append(t) for t in ts]
        [is_stimd.append(st) for st in stimd]
        [is_sig.append(sg) for sg in sig]
        [coms.append(co) for co in com]
        [peak_ts.append(pt) for pt in peak]
            
    is_stimd = np.array(is_stimd).flatten()
    is_sig   = np.array(is_sig).flatten()
    peak_ts  = np.array(peak_ts).flatten()
    coms     = np.array(coms).flatten()

    # interpolate to put everyone on the exact same time axis (small diffs in frame rate are possible)
    # start by aligning all time axes to zero and taking the mode of each bin
    nt_pre  = list()
    nt_post = list()
    fdur    = list()
    for taxis in t_axes:
        nt_pre.append(np.size(taxis[taxis<0]))
        nt_post.append(np.size(taxis[taxis>=0]))
        fdur.append(scipy.stats.mode(np.diff(taxis)))
    
    # base time axis making sure to include t = 0
    fdur, _    = scipy.stats.mode(np.array(fdur).flatten())
    nt_pre, _  = scipy.stats.mode(np.array(nt_pre).flatten()) 
    nt_post, _ = scipy.stats.mode(np.array(nt_post).flatten())   
    pre_t      = np.arange(-nt_pre*fdur,0,fdur)
    post_t     = np.arange(0,nt_post*fdur,fdur) 
    base_taxis = np.concatenate((pre_t,post_t)) 
    
    # convert all axes to base (mostly expected to be unchanged)
    for iResp in range(len(t_axes)):
        avg_resps[iResp] = np.interp(base_taxis,t_axes[iResp],avg_resps[iResp])  
        sem_resps[iResp] = np.interp(base_taxis,t_axes[iResp],sem_resps[iResp])    
        
    # convert from list to matrix if desired    
    if as_matrix:
        avgs = np.zeros((len(t_axes),len(base_taxis)))
        sems = np.zeros((len(t_axes),len(base_taxis)))
        for iResp in range(len(t_axes)):
            avgs[iResp,:] = avg_resps[iResp]
            sems[iResp,:] = sem_resps[iResp]
            
        # make sure NaN frames for shuttered pmt are the same
        nan_idx1    = np.argwhere(np.isnan(np.sum(avgs,axis=0))).flatten()
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
                    'response_type'  : resp_type, 
                    'experiment_type': expt_type, 
                    'which_neurons'  : which_neurons,
                    'analysis_params': deepcopy(params)
                    }
    
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    return summary_data

# ---------------
# %% get response stats (peak, dist from stim etc) for an area and experiment type
def get_full_resp_stats(area, params=params, expt_type='standard', resp_type='dff', eg_ids=None, which_neurons='non_stimd'):
    
    start_time      = time.time()
    print('Fetching opto-triggered response stats...')
    
    # get relevant keys
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    
    # restrict to only desired rec/stim if applicable
    if eg_ids is not None:
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
    dist_bins      = params['dist_bins_resp_prob']
    num_bins       = len(dist_bins)-1
    
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
            stimd, com, maxmin, peakt, trought, dist, sid = (twop_opto_analysis.TrigDffTrialAvg & this_key & 'is_stimd=1').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id')
            n    = len(stimd)
        else:
            sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')
            idx  = np.argwhere(np.array(incl)==1).flatten()
            n    = np.size(idx)
            keys = list(np.array(keys)[idx])
            sig  = list(np.array(sig)[idx])

            if which_neurons == 'all':
                stimd, com, maxmin, peakt, trought, dist, sid = (twop_opto_analysis.TrigDffTrialAvg & keys).fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id')
            elif which_neurons == 'non_stimd':
                stimd, com, maxmin, peakt, trought, dist, sid = (twop_opto_analysis.TrigDffTrialAvg & keys & 'is_stimd=0').fetch('is_stimd', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'min_dist_from_stim_um', 'stim_id')
            else:
                print('Unknown category of which_neurons, returning nothing')
                return None

        # pick trough or peak time, whichever is higher magnitude
        maxmin_t = list()
        for iNeuron in range(len(trought)):
            if maxmin[iNeuron] < 0:
                maxmin_t.append(trought[iNeuron])
            else:
                maxmin_t.append(peakt[iNeuron])

        # flatten list
        [is_stimd.append(st) for st in stimd]
        [is_sig.append(sg) for sg in sig]
        [coms.append(co) for co in com]
        [maxmins.append(mm) for mm in maxmin]
        [maxmins_ts.append(mmt) for mmt in maxmin_t]
        [dists.append(ds) for ds in dist]

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
                    'analysis_params'       : deepcopy(params)
                    }
    
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    return summary_data

# ---------------
# %% compare general responses stats between areas
def compare_response_stats(params=params, expt_type='standard', resp_type='dff', which_neurons='non_stimd', v1_data=None, m2_data=None):
    
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
    response_stats['response_magnitude_pval'], response_stats['response_magnitude_test_name'] = \
        general_stats.two_group_comparison(v1_data['max_or_min_vals'], m2_data['max_or_min_vals'], is_paired=False, tail="two-sided")
    response_stats['response_time_pval'], response_stats['response_time_test_name'] = \
        general_stats.two_group_comparison(v1_data['max_or_min_times_sec'], m2_data['max_or_min_times_sec'], is_paired=False, tail="two-sided")
    response_stats['response_com_pval'], response_stats['response_com_test_name'] = \
        general_stats.two_group_comparison(v1_data['com_sec'], m2_data['com_sec'], is_paired=False, tail="two-sided")

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
# %% plot comparison of overall proportion of significantly responding neurons
def plot_response_stats_comparison(params=params, expt_type='standard', resp_type='dff', which_neurons='non_stimd', response_stats=None, axis_handle=None, plot_what='response_magnitude'):
    
    # call this one dedicated function if it's just overall proportion (there for historical reasons)
    if plot_what == 'response_probability':
        response_stats, ax = plot_prop_response_comparison(params=params, expt_type=expt_type, resp_type=resp_type, axis_handle=axis_handle)
        return response_stats, ax
    
    # get data
    if response_stats is None:
        response_stats = compare_response_stats(params=params, expt_type=expt_type, resp_type=resp_type, which_neurons=which_neurons)
            
    # isolate desired variables    
    if plot_what == 'response_magnitude':
        v1_data  = response_stats['V1_stats']['max_or_min_vals']
        m2_data  = response_stats['M2_stats']['max_or_min_vals']
        pval     = response_stats[plot_what+'_pval']
        stats    = None
        histbins = params[plot_what+'_bins']
        xlbl     = 'Response magnitude (z-score)'
        ylbl     = 'Prop. of responding neurons'
        
    elif plot_what == 'response_time':
        v1_data  = response_stats['V1_stats']['max_or_min_times_sec']
        m2_data  = response_stats['M2_stats']['max_or_min_times_sec']
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
    
    start_time      = time.time()
    print('Fetching opto-triggered trials...')
    
    # get relevant keys
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    
    # restrict to only desired rec/stim if applicable
    if eg_ids is not None:
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
            tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (twop_opto_analysis.TrigDffTrial & this_key & 'is_stimd=1').fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id')
        elif which_neurons == 'non_stimd':
            sig, incl, keys = (twop_opto_analysis.TrigDffTrialAvgInclusion & this_key).fetch('is_significant', 'is_included', 'KEY')
            idx  = np.argwhere(np.array(incl)==1).flatten()
            keys = list(np.array(keys)[idx])
            sig  = list(np.array(sig)[idx])
            
            if signif_only:
                idx  = np.argwhere(np.array(sig)==1).flatten()
                keys = list(np.array(keys)[idx])
                sig  = list(np.array(sig)[idx]) 
            
            tids, trials, ts, sids, com, maxmin, peakt, trought, rids = (twop_opto_analysis.TrigDffTrial & keys & 'is_stimd=0').fetch('trial_id', 'trig_dff', 'time_axis_sec', 'stim_id', 'center_of_mass_sec_poststim', 'max_or_min_dff', 'time_of_peak_sec_poststim', 'time_of_trough_sec_poststim', 'roi_id')
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
# %% get single-trial opto-triggered responses for an area and experiment type
def xval_trial_data(area='M2', params=params, expt_type='high_trial_count', resp_type='dff', signif_only=True, which_neurons='non_stimd', trial_data=None, rng=None):

    # set random seed and delete low /  high tau values if applicable
    start_time      = time.time()
    print('Cross-validating reponse timing...')
    if rng is None:
        rng = np.random.default_rng(seed=params['random_seed'])
        
    # get data if necessary
    if trial_data is None:    
        trial_data = get_single_trial_data(area=area, params=params, expt_type=expt_type, resp_type=resp_type, signif_only=signif_only, which_neurons=which_neurons, relax_timing_criteria=params['xval_relax_timing_criteria'])
        
    # loop through rois and stims    
    roi_halves_r = list()
    roi_halves_p = list()
    median_half1 = list()
    median_half2 = list()
    unique_rois  = list(np.unique(trial_data['roi_ids']))
    for roi in unique_rois:
        ridx         = trial_data['roi_ids']==roi
        these_trials = trial_data['trial_ids'][ridx]
        these_stims  = trial_data['stim_ids'][ridx]
        coms         = trial_data['com_sec'][ridx]
        peaks        = trial_data['peak_or_trough_time_sec'][ridx]
        trial_dffs   = trial_data['trig_dff_trials'][ridx]
        t_axes       = trial_data['time_axis_sec'][ridx]
        unique_stims = list(np.unique(these_stims))
        
        # take random halves of trials and compare timing stats for each 
        for stim in unique_stims:
            tidx        = these_trials[these_stims==stim] 
            tidx_shuff  = np.deepcopy(tidx)
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
                        else:
                            timing_set1[iShuff] = trough1
                            timing_set2[iShuff] = trough2
                        
                    elif params['xval_timing_metric'] == 'com':
                        timing_set1[iShuff] = com1
                        timing_set2[iShuff] = com2
                        
                    else:
                        print('unknown parameter value for timing metric, returning nothing')
                        return None
                
                # or just take a median of pre-computed trial-by-trial features    
                else:
                    if params['xval_timing_metric'] == 'peak':
                        timing_set1[iShuff] = np.nanmedian(peaks[half1])
                        timing_set2[iShuff] = np.nanmedian(peaks[half2])
                        
                    elif params['xval_timing_metric'] == 'com':
                        timing_set1[iShuff] = np.nanmedian(coms[half1])
                        timing_set2[iShuff] = np.nanmedian(coms[half2])
                        
                    else:
                        print('unknown parameter value for timing metric, returning nothing')
                        return None
            
            # collect correlation between metrics on halves and median metric
            cc, pval = scipy.stats.pearsonr(timing_set1, timing_set2) 
            roi_halves_r.append(cc)
            roi_halves_p.append(pval)
            median_half1.append(np.nanmedian(timing_set1))
            median_half2.append(np.nanmedian(timing_set2))
            
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    xval_results = {
                'timing_metric'     : params['xval_timing_metric'], 
                'trial_halves_cc'   : roi_halves_r, 
                'trial_halves_pval' : t_axes,
                'median_trialset1'  : median_half1,
                'median_trialset2'  : median_half2, 
                'response_type'     : resp_type, 
                'experiment_type'   : expt_type, 
                'which_neurons'     : which_neurons,
                'analysis_params'   : deepcopy(params)
                }
    
    
    return xval_results, trial_data
# ====================
# SANDBOX
# =====================


# %%
v1_avgs = get_avg_trig_responses('V1', params=params, expt_type='standard', resp_type='dff',signif_only=False)
m2_avgs = get_avg_trig_responses('M2', params=params, expt_type='standard', resp_type='dff',signif_only=False)

idx = np.argsort(v1_avgs['peak_times_sec']).flatten()
resp_mat_v1 = v1_avgs['trig_dff_avgs'][idx,:]
idx = np.argsort(m2_avgs['peak_times_sec'])
resp_mat_m2 = m2_avgs['trig_dff_avgs'][idx,:]
maxval = np.nanmax(np.abs(resp_mat_m2))
num_v1 = np.size(resp_mat_v1,axis=0)
num_m2 = np.size(resp_mat_m2,axis=0)

for iNeuron in range(num_v1):
    resp_mat_v1[iNeuron,:] = resp_mat_v1[iNeuron,:]/np.nanmax(resp_mat_v1[iNeuron,:]) #v1_mat[iNeuron,idx_v1]
for iNeuron in range(num_m2):
    resp_mat_m2[iNeuron,:] = resp_mat_m2[iNeuron,:]/np.nanmax(resp_mat_m2[iNeuron,:]) #m2_mat[iNeuron,idx_m2]    
    
t_axis_v1   = v1_avgs['time_axis_sec']
t_axis_m2   = m2_avgs['time_axis_sec']
# 
plt.matshow(resp_mat_v1,cmap='coolwarm',vmin=-1,vmax=1)
plt.colorbar()
plt.matshow(resp_mat_m2,cmap='coolwarm',vmin=-1,vmax=1)
plt.colorbar()
# %%
# v1_mat = v1_avgs['trig_dff_avgs']
# m2_mat = m2_avgs['trig_dff_avgs']

# idx_v1 = np.argwhere(np.isnan(np.sum(v1_mat,axis=0)))[-1]+1
# idx_m2 = np.argwhere(np.isnan(np.sum(m2_mat,axis=0)))[-1]+1



# %%
v1_avg = np.nanmean(resp_mat_v1,axis=0)
v1_sem = np.nanstd(resp_mat_v1,axis=0)/np.sqrt(num_v1-1)
m2_avg = np.nanmean(resp_mat_m2,axis=0)
m2_sem = np.nanstd(resp_mat_m2,axis=0)/np.sqrt(num_m2-1)
t_axis_v1   = v1_avgs['time_axis_sec']
t_axis_m2   = m2_avgs['time_axis_sec']

plt.plot(t_axis_v1,v1_avg,'-',color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'])
plt.plot(t_axis_m2,m2_avg,'-',color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'])
# %%
plot_prop_response_comparison(resp_type='deconv')
    
# TO DO

# seuqence xval
# PCA
# opto vs tau, including dyanmics of that
# %%

# %%
