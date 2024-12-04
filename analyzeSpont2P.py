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
    
    return taus, keys, np.size(is_good)
# %%
# ---------------
# statistically compare taus across areas and plot
def plot_area_tau_comp(params=params, dff_type='residuals_dff', bin_size=.2, max_tau=10, axisHandle=None, v1_taus=None, m2_taus=None):

    # get taus
    if v1_taus is None:
        v1_taus, _ = get_all_tau('V1',params=params,dff_type=dff_type)
    if m2_taus is None:
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

    histbins     = np.arange(0,max_tau,bin_size)
    v1_counts, _ = np.histogram(v1_taus,bins=histbins,density=False)
    m2_counts, _ = np.histogram(m2_taus,bins=histbins,density=False)
    xaxis        = histbins[:-1]+bin_size/2
    ax.plot(xaxis,np.cumsum(v1_counts)/np.sum(v1_counts),color=params['V1_cl'],label=params['V1_lbl'])
    ax.plot(xaxis,np.cumsum(m2_counts)/np.sum(m2_counts),color=params['M2_cl'],label=params['M2_lbl'])
    ax.plot(tau_stats['V1_median'],0.03,'v',color=params['V1_cl'])
    ax.plot(tau_stats['M2_median'],0.03,'v',color=params['M2_cl'])
    ax.text(.1,.8,'p = {:1.2f}'.format(tau_stats['pval']))

    ax.set_xscale('log')
    ax.set_xlabel('$\\tau$ (sec)')
    ax.set_ylabel('Prop. neurons')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return tau_stats, ax 

# %%
# ---------------
# get centroid and sess info for a list of tau keys
def get_centroids_by_rec(tau_keys):

    # find unique sessions 
    subj_date_list = ['{}-{}'.format(this_key['subject_fullname'],this_key['session_date']) for this_key in tau_keys] 
    unique_sess    = np.unique(subj_date_list)

    # get centroids and assign numeric session ids
    centroids = list()
    sess_ids  = list()
    for idx, this_key in tau_keys:
        centroids.append((VM['twophoton'].Roi2P & this_key).fetch1('centroid_pxls'))
        sess_ids.append(unique_sess.index(subj_date_list[idx]))
    
    return np.array(centroids), np.array(sess_ids) 

# %%
# ---------------
# statistically compare taus across areas and plot
def clustering_by_tau(taus, centroids, rec_ids, params=params):

    # pairwise dists and tau diffs (per rec of course)
    unique_recs = np.unique(rec_ids).aslist()
    roi_dists   = list()
    tau_diffs   = list()
    for rec in unique_recs:
        idx = rec_ids == rec
        sub_centroids = centroids[idx]
        sub_taus      = taus[idx]
        num_cells     = np.sum(idx==1)
        
        for iCell1 in range(num_cells):
            for iCell2 in range(num_cells):
                if iCell1 < iCell2:
                    this_dist = np.sqrt((sub_centroids[iCell1][0] - sub_centroids[iCell2][0])**2+(sub_centroids[iCell1][1] - sub_centroids[iCell2][1])**2)
                    roi_dists.append(this_dist)
                    tau_diffs.append(np.abs(sub_taus[iCell1] - sub_taus[iCell2]))
    
    roi_dists = np.array(roi_dists)
    tau_diffs = np.array(tau_diffs)
    
    # bin, boostrap
    clust_results = dict()

    return clust_results 


# %%
v1_taus, v1_keys, v1_total = get_all_tau('V1', params = params, dff_type = 'residuals_dff')
m2_taus, m2_keys, m2_total = get_all_tau('M2', params = params, dff_type = 'residuals_dff')
# 
tau_stats, _ = plot_area_tau_comp(v1_taus=v1_taus, m2_taus=m2_taus)
# %%
