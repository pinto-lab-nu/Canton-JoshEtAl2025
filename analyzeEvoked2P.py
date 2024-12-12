# ========================================
# =============== SET UP =================
# ========================================

# %% import stuff
from utils import connect_to_dj
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
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
        'trigdff_param_set_id_dff'       : 4, 
        'trigdff_param_set_id_deconv'    : 5, 
        'trigdff_inclusion_param_set_id' : 3,
        'trigdff_inclusion_param_set_id_notiming' : 4, # for inhibition, we may want to relax trough timing constraint
        'trigspeed_param_set_id'         : 1,
        'prop_resp_bins'                 : np.arange(0,.41,.01),
        }

params['general_params'] = deepcopy(tau_params['general_params'])

# ========================================
# =============== METHODS ================
# ========================================

# %%
# ---------------
# get list of dj keys for sessions of a certain type
# 'standard' is single neurons with <= 10 trials & 5 spirals, 'short_stim' is single neurons <= 10 trials & < 5 spirals, 
# 'high_trial_count' is single neurons with > 10 trials, 'multi_cell' has at least one group with mutiple stim'd neurons
def get_keys_for_expt_types(area, params=params, expt_type='standard'):
    
    # get primary keys for query
    mice              = params['general_params']['{}_mice'.format(area)]
    
    # get relavant keys 
    keys = list()
    for mouse in mice:
        opto_data = (twop_opto_analysis.Opto2PSummary & {'subject_fullname': mouse}).fetch(as_dict=True)
        for this_sess in opto_data:
            has_multicell_stim = bool(np.sum(np.array(this_sess['num_cells_per_stim'])>1)>1)
            max_num_trials     = np.max(np.array(this_sess['num_trials_per_stim']))
            max_dur            = np.max(np.array(this_sess['dur_sec_per_stim']))
            
            # choose experiments that match type
            if expt_type == 'standard':
                if np.logical_and((not has_multicell_stim), np.logical_and(max_num_trials <= 10 , max_dur >= 0.2)):
                    keys.append(this_sess)
            elif expt_type == 'short_stim':
                if np.logical_and((not has_multicell_stim), np.logical_and(max_num_trials <= 10 , max_dur < 0.2)):
                    keys.append(this_sess)
            elif expt_type == 'high_trial_count':
                if np.logical_and((not has_multicell_stim) , max_num_trials > 10):
                    keys.append(this_sess)
            elif expt_type == 'multi_cell':
                if has_multicell_stim:
                    keys.append(this_sess)
            else:
                print('Unknown experiment type, doing nothing')
                return
    
    return keys

# %%
# get proportion of significantly responding neurons for an area and experiment type
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
                    }
    
    return summary_data
        
# %%
# plot comparison of proportion of significantly responding neurons
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
    
# %%
# get average opto-triiggered responses for an area and experiment type
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
                return

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
                    }
    
    end_time = time.time()
    print("     done after {: 1.2f} min".format((end_time-start_time)/60))
    
    return summary_data

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
# basic triggered stats -- incidence, magnitude, fov egs, by space etc
# eg avg trig responses. all time courses in a field of view + peak + time of peak
# time course -- summary and fov sequence heatmaps
# for each fov, Compare tau of post stim decay to predicted tau from eigenvalue of xcorr mat
# trig running
# opsin expression
# seuqence xval
# PCA
# opto vs tau, including dyanmics of that
# %%
