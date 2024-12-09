# %% import stuff
from utils import connect_to_dj
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from copy import deepcopy
from schemas import spont_timescales
from utils.stats import general_stats
from utils.plotting import plot_pval_circles
from utils.plotting import plot_fov_heatmap

import time

# connect to dj 
VM     = connect_to_dj.get_virtual_modules()

# %% declare default params 
params = {
        'random_seed'               : 42, 
        'max_tau'                   : None,
        'tau_hist_bins'             : np.arange(0,10.2,.2),
        'clustering_dist_bins'      : np.arange(0,350,50),
        'clustering_num_boot_iter'  : 10000,
        'clustering_num_shuffles'   : 1000,
        'clustering_zscore_taus'    : True,
        }
params['general_params'] = {
        'V1_mice'     : ['jec822_NCCR62','jec822_NCCR63','jec822_NCCR66','jec822_NCCR86'] ,
        'M2_mice'     : ['jec822_NCCR32','jec822_NCCR72','jec822_NCCR73','jec822_NCCR77'] ,
        'V1_cl'       : np.array([180, 137, 50])/255 ,#np.array([120, 120, 120])/255 ,
        'M2_cl'       : np.array([43, 67, 121])/255 ,#np.array([90, 60, 172])/255 ,
        'V1_lbl'      : 'VISp' ,
        'M2_lbl'      : 'MOs' ,
        'corr_param_id_noGlm_dff'        : 2, 
        'corr_param_id_residuals_dff'    : 5,
        'corr_param_id_residuals_deconv' : 4, 
        'twop_inclusion_param_set_id'    : 4,
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
    mice              = params['general_params']['{}_mice'.format(area)]
    corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
    
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
    
    # figure out how many of those are somas
    seg_keys   = VM['twophoton'].Segmentation2P & keys
    is_soma    = np.array((VM['twophoton'].Roi2P & seg_keys).fetch('is_soma'))
    total_soma = np.sum(is_soma)
    
    end_time = time.time()
    print("     done after {: 1.1g} min".format((end_time-start_time)/60))
    
    return taus, keys, total_soma
# %%
# ---------------
# statistically compare taus across areas and plot
def plot_area_tau_comp(params=params, dff_type='residuals_dff', axis_handle=None, v1_taus=None, m2_taus=None):

    # get taus
    if v1_taus is None:
        v1_taus, _ , _ = get_all_tau('V1',params=params,dff_type=dff_type)
    if m2_taus is None:
        m2_taus, _ , _ = get_all_tau('M2',params=params,dff_type=dff_type)
    
    # compute stats
    tau_stats = dict()
    tau_stats['V1_num_cells'] = np.size(v1_taus)
    tau_stats['V1_mean']      = np.mean(v1_taus)
    tau_stats['V1_sem']       = np.std(v1_taus,ddof=1) / np.sqrt(tau_stats['V1_num_cells']-1)
    tau_stats['V1_median']    = np.median(v1_taus)
    tau_stats['V1_iqr']       = scipy.stats.iqr(v1_taus)
    tau_stats['M2_num_cells'] = np.size(m2_taus)
    tau_stats['M2_mean']      = np.mean(m2_taus)
    tau_stats['M2_sem']       = np.std(m2_taus,ddof=1) / np.sqrt(tau_stats['M2_num_cells']-1)
    tau_stats['M2_median']    = np.median(m2_taus)
    tau_stats['M2_iqr']       = scipy.stats.iqr(m2_taus)
    tau_stats['pval'], tau_stats['test_name'] = general_stats.two_group_comparison(v1_taus, m2_taus, is_paired=False, tail="two-sided")

    # plot
    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle

    histbins     = params['tau_hist_bins']
    v1_counts, _ = np.histogram(v1_taus,bins=histbins,density=False)
    m2_counts, _ = np.histogram(m2_taus,bins=histbins,density=False)
    xaxis        = histbins[:-1]+np.diff(histbins)[0]
    ax.plot(xaxis,np.cumsum(v1_counts)/np.sum(v1_counts),color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'])
    ax.plot(xaxis,np.cumsum(m2_counts)/np.sum(m2_counts),color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'])
    ax.plot(tau_stats['V1_median'],0.03,'v',color=params['general_params']['V1_cl'])
    ax.plot(tau_stats['M2_median'],0.03,'v',color=params['general_params']['M2_cl'])
    ax.text(.18,.8,'p = {:1.2g}'.format(tau_stats['pval']))

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
    unique_sess    = np.unique(subj_date_list).tolist()

    # get centroids and assign numeric session ids
    centroids = list()
    sess_ids  = list()
    for idx, this_key in enumerate(tau_keys):
        centr    = (VM['twophoton'].Roi2P & this_key).fetch1('centroid_pxls')
        xs, ys   = (VM['twophoton'].Scan & this_key).fetch1('microns_per_pxl_x','microns_per_pxl_y')
        centroids.append(np.array([centr[0,0]*ys, centr[0,1]*xs]))
        sess_ids.append(unique_sess.index(subj_date_list[idx]))
    
    return np.array(centroids), np.array(sess_ids) 

# %%
# ---------------
# statistically compare taus across areas and plot
def clustering_by_tau(taus, centroids, rec_ids, params=params, rng=None):

    # set random seed and delete very high tau values if applicable
    start_time      = time.time()
    print('Performing clustering analysis...')
    if rng is None:
        rng = np.random.default_rng(seed=params['random_seed'])
    
    if params['max_tau'] is not None:
        del_idx   = np.argwhere(taus>params['max_tau'])
        taus      = np.delete(taus,del_idx)
        centroids = np.delete(centroids,del_idx,axis=0)
        rec_ids   = np.delete(rec_ids,del_idx)
    
    # pairwise dists and tau diffs (per rec of course)
    unique_recs = np.unique(rec_ids).tolist()
    roi_dists   = list()
    tau_diffs   = list()
    for rec in unique_recs:
        idx           = rec_ids == rec
        sub_centroids = centroids[idx]
        sub_taus      = taus[idx]
        num_cells     = np.sum(idx==1)
        if params['clustering_zscore_taus']:
            sub_taus = (sub_taus - np.mean(sub_taus))/np.std(sub_taus)
        
        for iCell1 in range(num_cells):
            for iCell2 in range(num_cells):
                if iCell1 < iCell2:
                    centr1    = sub_centroids[iCell1].flatten()
                    centr2    = sub_centroids[iCell2].flatten()
                    this_dist = np.sqrt((centr1[0] - centr2[0])**2 + (centr1[1] - centr2[1])**2)
                    roi_dists.append(this_dist)
                    tau_diffs.append(np.abs(sub_taus[iCell1] - sub_taus[iCell2]))
    
    roi_dists = np.array(roi_dists).flatten()
    tau_diffs = np.array(tau_diffs).flatten()
    
    # boostrap and do average tau diff per dist bin
    clust_results = dict()
    num_iter      = params['clustering_num_boot_iter']
    bins          = params['clustering_dist_bins']
    num_bins      = np.size(bins)-1
    tau_mat       = np.zeros((num_iter,num_bins)) + np.nan
    num_cells     = np.size(tau_diffs)
    idx_vec       = np.arange(num_cells)
    
    for iBoot in range(num_iter):
        idx       = rng.choice(idx_vec,num_cells)
        this_dist = roi_dists[idx]
        this_tau  = tau_diffs[idx]
        
        for iBin in range(num_bins):
            sub_idx = np.logical_and(this_dist >= bins[iBin] , this_dist < bins[iBin+1])
            if np.sum(sub_idx) > 0:
                tau_mat[iBoot,iBin] = np.mean(this_tau[sub_idx])

    # collect some results
    clust_results['dist_um']       = bins[:-1]+np.diff(bins)[0]/2
    clust_results['num_boot_iter'] = num_iter  
    clust_results['tau_diff_bydist_mean'] = np.nanmean(tau_mat,axis=0).flatten()
    clust_results['tau_diff_bydist_std']  = np.nanstd(tau_mat,axis=0,ddof=1).flatten()
    
    # shuffles 
    num_shuff    = params['clustering_num_shuffles']
    shuffle_mat  = np.zeros((num_shuff,num_bins)) + np.nan
    dist_shuffle = deepcopy(roi_dists)
    
    for iShuff in range(num_shuff):
        rng.shuffle(dist_shuffle)
        for iBin in range(num_bins):
            sub_idx = np.logical_and(dist_shuffle >= bins[iBin] , dist_shuffle < bins[iBin+1])
            if np.sum(sub_idx) > 0:
                shuffle_mat[iShuff,iBin] = np.mean(tau_diffs[sub_idx])
            
    # collect results and do stats
    shuffle_mean = np.nanmean(shuffle_mat,axis=0).flatten()
    clust_results['tau_diff_bydist_shuffle_mean'] = shuffle_mean
    clust_results['tau_diff_bydist_shuffle_std']  = np.nanstd(shuffle_mat,axis=0,ddof=1).flatten()
    pvals = np.zeros(num_bins)
    for iBin in range(num_bins):
        if clust_results['tau_diff_bydist_mean'][iBin] <= shuffle_mean[iBin]:
            pvals[iBin] = np.sum(tau_mat[:,iBin] > shuffle_mean[iBin]) / num_iter
        else:
            pvals[iBin] = np.sum(tau_mat[:,iBin] < shuffle_mean[iBin]) / num_iter
    clust_results['tau_diff_bydist_pvals'] = pvals
    
    # FDR correction
    clust_results['tau_diff_bydist_isSig'], _ = general_stats.FDR(clust_results['tau_diff_bydist_pvals'])
    
    end_time = time.time()
    print("     done after {: 1.2f} sec".format((end_time-start_time)))
            
    return clust_results, tau_mat

# %%
# ---------------
# statistically compare taus across areas and plot
def plot_clustering_comp(v1_clust=None, m2_clust=None, params=params, axis_handle=None):

    # plot
    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle

    # data
    xaxis   = v1_clust['dist_um']
    v1_mean = v1_clust['tau_diff_bydist_mean'] - v1_clust['tau_diff_bydist_shuffle_mean']
    v1_std  = v1_clust['tau_diff_bydist_std'] 
    m2_mean = m2_clust['tau_diff_bydist_mean'] - m2_clust['tau_diff_bydist_shuffle_mean']
    m2_std  = m2_clust['tau_diff_bydist_std'] 
    ax.errorbar(x=xaxis,y=v1_mean,yerr=v1_std,color=params['general_params']['V1_cl'],label=params['general_params']['V1_lbl'],marker='.')
    ax.errorbar(x=xaxis,y=m2_mean,yerr=m2_std,color=params['general_params']['M2_cl'],label=params['general_params']['M2_lbl'],marker='.')
    ax.plot(np.array([0,xaxis[-1]+xaxis[0]]),np.array([0,0]),'--',color='gray')
    
    #  pvals
    plot_pval_circles(ax, xaxis, v1_mean-v1_std, v1_clust['tau_diff_bydist_pvals'], 
                    where='below',color=params['general_params']['V1_cl'],isSig=v1_clust['tau_diff_bydist_isSig'])
    plot_pval_circles(ax, xaxis, m2_mean-m2_std-0.01, m2_clust['tau_diff_bydist_pvals'], 
                    where='below',color=params['general_params']['M2_cl'],isSig=m2_clust['tau_diff_bydist_isSig'])

    if params['clustering_zscore_taus']:
        ax.set_ylabel('|$\\tau$ diff (z-score)| - shuffle')
    else:
        ax.set_ylabel('|$\\tau$ diff (sec)| - shuffle')
    ax.set_xlabel('Pairwise dist. $\\mu$m')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax 

# %%
# ---------------
# statistically compare taus across areas and plot
def plot_tau_fov(tau_keys, sess_ids, which_sess=0, do_zscore=params['clustering_zscore_taus'], prctile_cap=[0,95], axis_handle=None, fig_handle=None):

    # fetch roi coordinates for desired session
    idx  = np.argwhere(sess_ids == which_sess)
    keys = np.array(tau_keys)[idx].tolist()
    
    taus                         = (spont_timescales.TwopTau & keys).fetch('tau')
    row_pxls, col_pxls           = (VM['twophoton'].Roi2P & keys).fetch('row_pxls','col_pxls')
    num_rows,num_cols,um_per_pxl = (VM['twophoton'].Scan & keys[0]).fetch1('lines_per_frame','pixels_per_line','microns_per_pxl_y')
    roi_coords = [row_pxls,col_pxls]
    im_size    = (num_rows,num_cols)
    
    if do_zscore:
        taus = np.array(taus)
        taus = ((taus-np.mean(taus))/np.std(taus)).tolist()
        lbl  = '$\\tau$ (z-score)'
    else:
        lbl  = '$\\tau$ (sec)'
    
    # send to generic function
    ax, fig = plot_fov_heatmap(roi_vals=taus, roi_coords=roi_coords, im_size=im_size, um_per_pxl=um_per_pxl, \
                              prctile_cap=prctile_cap, cbar_lbl=lbl, axisHandle=axis_handle, figHandle=fig_handle)
    
    return ax, fig

# # %%
# v1_taus, v1_keys, v1_total = get_all_tau('V1', params = params, dff_type = 'residuals_dff')
# m2_taus, m2_keys, m2_total = get_all_tau('M2', params = params, dff_type = 'residuals_dff')
# # %%
# tau_stats, _ = plot_area_tau_comp(v1_taus=v1_taus, m2_taus=m2_taus)
# tau_stats
# # %%
# v1_centr, v1_rec_ids = get_centroids_by_rec(v1_keys)
# m2_centr, m2_rec_ids = get_centroids_by_rec(m2_keys)

# # %% fig 2g: clustering
# these_params = deepcopy(params)
# these_params['random_seed'] = 42
# these_params['max_tau'] = None
# these_params['clustering_num_boot_iter'] = 1000
# these_params['clustering_dist_bins'] = np.arange(0,330,30)
# these_params['clustering_zscore_taus'] = True

# # rng = np.random.default_rng(seed=these_params['random_seed'])

# # clust_stats_v1 , tau_diff_mat_v1 = clustering_by_tau(v1_taus, v1_centr, v1_rec_ids, these_params,rng=rng)
# # clust_stats_m2 , tau_diff_mat_m2 = clustering_by_tau(m2_taus, m2_centr, m2_rec_ids, these_params,rng=rng)
# cax = plot_clustering_comp(v1_clust=clust_stats_v1,m2_clust=clust_stats_m2, params=these_params)
# # %%
# im_ax = plot_tau_fov(v1_keys, v1_rec_ids, which_sess=1, do_zscore=False, prctile_cap=[0,95])
# # good ones m2: 0, 4, 5, 10 (95th prct), 17, 22
# # good ones v1: 0 , 2 
# # try just long or short timescale cells for clusetring
# # SEPARATE TAU FROM GENERIC FOV PLOT FOR REUSE
# # %%
