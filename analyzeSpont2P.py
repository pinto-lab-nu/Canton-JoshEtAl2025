# %% import stuff
from utils import connect_to_dj
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from copy import deepcopy
from schemas import spont_timescales
from utils.stats import general_stats
import time

# %% connect to dj and declare default params
VM     = connect_to_dj.get_virtual_modules()
params = {
        'V1_mice' : ['jec822_NCCR62','jec822_NCCR63','jec822_NCCR66','jec822_NCCR86'] ,
        'M2_mice' : ['jec822_NCCR32','jec822_NCCR72','jec822_NCCR73','jec822_NCCR77'] ,
        'V1_cl'   : np.array([120, 120, 120])/255 ,
        'M2_cl'   : np.array([90, 60, 172])/255 ,
        'V1_lbl'  : 'VISp' ,
        'M2_lbl'  : 'MOs' ,
        'corr_param_id_noGlm_dff'        : 2, 
        'corr_param_id_residuals_dff'    : 5, 
        'corr_param_id_residuals_deconv' : 4, 
        'twop_inclusion_param_set_id'    : 2,
        'tau_param_set_id'               : 1,
        }

# FUNCTIONS
# %%
# ---------------
# retrieve all taus for a given area and set of parameters, from dj database
def get_all_tau(area, params = params, dff_type = 'residuals_dff'):
    
    start_time      = time.time()
    print('Fetching all taus for {}...'.format(area))
        
    # get primary keys for query
    mice              = params['{}_mice'.format(area)]
    corr_param_set_id = params['corr_param_id_{}'.format(dff_type)]
    
    # get relavant keys 
    keys = list()
    for mouse in mice:
        primary_key = {'subject_fullname': mouse, 'corr_param_set_id': corr_param_set_id, 'tau_param_set_id': params['tau_param_set_id']}
        keys.append((spont_timescales.TwopTau & primary_key).fetch('KEY'))
    keys    = [entries for subkeys in keys for entries in subkeys] # flatten
    
    # retrieve taus with inclusion flags, return just good ones (with corresponding keys)
    taus    = np.array((spont_timescales.TwopTau & keys).fetch('tau'))
    is_good = np.array((spont_timescales.TwopTauInclusion & keys & 'twop_inclusion_param_set_id={}'.format(params['twop_inclusion_param_set_id'])).fetch('is_good_tau_roi'))
    
    taus = taus[is_good==1]
    keys = list(np.array(keys)[is_good==1])
    
    end_time = time.time()
    print("     done after {: 1.1g} min".format((end_time-start_time)/60))
    
    return taus, keys
# %%
# ---------------
# statistically compare taus across areas and plot
def plot_area_tau_comp(params=params, dff_type='residuals_dff', bin_size=.2, axisHandle=None):

    # get taus
    v1_taus, _ = get_all_tau('V1',params=params,dff_type=dff_type)
    m2_taus, _ = get_all_tau('M2',params=params,dff_type=dff_type)
    
    # compute stats
    tau_stats = dict()
    tau_stats['V1_num_cells'] = np.size(v1_taus)
    tau_stats['V1_mean']      = np.mean(v1_taus)
    tau_stats['V1_sem']       = np.std(v1_taus,ddof=1) / np.sqrt(tau_stats['V1_num_cells'])
    tau_stats['V1_median']    = np.median(v1_taus)
    tau_stats['V1_iqr']       = scipy.stats.iqr(v1_taus)
    tau_stats['M2_num_cells'] = np.size(m2_taus)
    tau_stats['M2_mean']      = np.mean(m2_taus)
    tau_stats['M2_sem']       = np.std(m2_taus,ddof=1) / np.sqrt(tau_stats['M2_num_cells'])
    tau_stats['M2_median']    = np.median(m2_taus)
    tau_stats['M2_iqr']       = scipy.stats.iqr(m2_taus)
    tau_stats['pval'], tau_stats['test_name'] = general_stats.two_group_comparison(v1_taus, m2_taus, is_paired=False, tail="two-sided")

    # plot
    if axisHandle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axisHandle

    histbins     = np.arange(0,30,bin_size)
    v1_counts, _ = np.histogram(v1_taus,bins=histbins,density=False)
    m2_counts, _ = np.histogram(m2_taus,bins=histbins,density=False)
    ax.plot(histbins[:-1]+bin_size/2,np.cumsum(v1_counts)/np.sum(v1_counts),color=params['V1_cl'],label=params['V1_lbl'])
    ax.plot(histbins[:-1]+bin_size/2,np.cumsum(m2_counts)/np.sum(m2_counts),color=params['M2_cl'],label=params['M2_lbl'])

    ax.set_xscale('log')
    ax.set_xlabel('$\\tau$ (sec)')
    ax.set_ylabel('Prop. neurons')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return tau_stats, ax 
    
    