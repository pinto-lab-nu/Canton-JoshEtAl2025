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
        'trigdff_param_set_id_dff'       : 1, 
        'trigdff_param_set_id_deconv'    : 3, 
        'trigdff_inclusion_param_set_id' : 1,
        'trigspeed_param_set_id'         : 1,
        'prop_resp_bins'                 : np.arange(0,.41,.01),
        }

params['general_params'] = deepcopy(tau_params['general_params'])

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
    
    expt_keys = get_keys_for_expt_types(area, params=params, expt_type=expt_type)
    
    trigdff_param_set_id           = params['trigdff_param_set_id_{}'.format(resp_type)]
    trigdff_inclusion_param_set_id = params['trigdff_inclusion_param_set_id']
    
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
plot_prop_response_comparison(resp_type='deconv')
    
# TO DO
# basic triggered stats -- incidence, magnitude, fov egs, by space etc
# time course
# trig running
# opsin expression
# seuqence xval
# PCA
# opto vs tau, including dyanmics of that
# %%
v1_data = get_prop_responding_neurons('V1')
v1_data
# %%
