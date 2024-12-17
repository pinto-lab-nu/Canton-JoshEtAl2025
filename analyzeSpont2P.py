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

import time

# connect to dj 
VM     = connect_to_dj.get_virtual_modules()

# %% declare default params 
params = {
        'random_seed'               : 42, 
        'max_tau'                   : None,
        'min_tau'                   : None,
        'tau_hist_bins'             : np.arange(0,10.2,.2),
        'tau_hist_bins_xcorr'       : np.arange(0,20.2,.2),
        'tau_hist_bins_eigen'       : np.arange(0,101,1),
        'clustering_dist_bins'      : np.arange(0,350,50),
        'clustering_num_boot_iter'  : 10000,
        'clustering_num_shuffles'   : 1000,
        'clustering_zscore_taus'    : True,
        }
params['general_params'] = {
        'V1_mice'     : ['jec822_NCCR62','jec822_NCCR63','jec822_NCCR66','jec822_NCCR86'] ,
        'M2_mice'     : ['jec822_NCCR32','jec822_NCCR72','jec822_NCCR73','jec822_NCCR77'] ,
        'V1_cl'       : np.array([180, 137, 50])/255 ,
        'M2_cl'       : np.array([43, 67, 121])/255 ,
        'V1_sh'       : np.array([200, 147, 60, 90])/255 ,
        'M2_sh'       : np.array([53, 47, 141, 90])/255 ,
        'V1_lbl'      : 'VISp' ,
        'M2_lbl'      : 'MOs' ,
        'corr_param_id_noGlm_dff'        : 2, 
        'corr_param_id_residuals_dff'    : 5,
        'corr_param_id_residuals_deconv' : 4, 
        'twop_inclusion_param_set_id'    : 4,
        'tau_param_set_id'               : 1,
        }

# ========================================
# =============== METHODS ================
# ========================================

# ---------------
# %% retrieve dj keys for unique experimental sessions given area and set of parameters
def get_single_sess_keys(area, params=params, dff_type='residuals_dff'):
    
    """
    get_single_sess_keys(area, params=params, dff_type='residuals_dff')
    returns a list of dj session keys for an area
    
    INPUT:
    area: 'V1' or 'M2'
    params: dictionary as the one on top of this file, used to determine
            mice to look for and which parameter sets correspond to desired
            dff_type
    dff_type: 'residuals_dff' for residuals of running linear regression (default)
              'residuals_deconv' for residuals of running Poisson GLM on deconvolved traces
              'noGlm_dff' for plain dff traces
    """
    
    # get primary keys for query
    mice              = params['general_params']['{}_mice'.format(area)]
    corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
    tau_param_set_id  = params['general_params']['tau_param_set_id']
    incl_set_id       = params['general_params']['twop_inclusion_param_set_id']
    
    # get relavant keys 
    keys = list()
    for mouse in mice:
        primary_key = {'subject_fullname': mouse, 'corr_param_set_id': corr_param_set_id, 'tau_param_set_id': tau_param_set_id, 'twop_inclusion_param_set_id': incl_set_id}
        mouse_keys  = (spont_timescales.TwopTauInclusion & primary_key & 'is_good_tau_roi=1').fetch('KEY')
        [keys.append(ikey) for ikey in mouse_keys]
        
    # find unique ones and rearrange    
    subj_date_list = ['{},{}'.format(this_key['subject_fullname'],this_key['session_date']) for this_key in keys] 
    unique_sess    = np.unique(subj_date_list).tolist()
    
    sess_keys = list()
    for sess in unique_sess:
        idx  = sess.find(',')
        subj = sess[:idx]
        dat  = sess[idx+1:]
        sess_keys.append({'subject_fullname':subj,'session_date':dat})
    
    return sess_keys

# ---------------
# %% retrieve all taus for a given area and set of parameters, from dj database
def get_all_tau(area, params = params, dff_type = 'residuals_dff'):
    
    """
    get_all_tau(area, params=params, dff_type='residuals_dff')
    fetches timescales for every neuron in a given area
    
    INPUT:
    area: 'V1' or 'M2'
    params: dictionary as the one on top of this file, used to determine
            mice to look for and which parameter sets correspond to desired
            dff_type
    dff_type: 'residuals_dff' for residuals of running linear regression (default)
              'residuals_deconv' for residuals of running Poisson GLM on deconvolved traces
              'noGlm_dff' for plain dff traces
              
    OUTPUT:
    taus: vector with taus that pass inclusion criteria 
          in params['general_params']['twop_inclusion_param_set_id']
    keys: list of keys corresponding to each tau
    total_soma: total number of somas regardless of inclusion criteria
    """
    
    start_time      = time.time()
    print('Fetching all taus for {}...'.format(area))
        
    # get primary keys for query
    mice              = params['general_params']['{}_mice'.format(area)]
    corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
    tau_param_set_id  = params['general_params']['tau_param_set_id']
    incl_set_id       = params['general_params']['twop_inclusion_param_set_id']
    
    # get relavant keys, filtering for inclusion for speed
    keys = list()
    for mouse in mice:
        primary_key = {'subject_fullname': mouse, 'corr_param_set_id': corr_param_set_id, 'tau_param_set_id': tau_param_set_id, 'twop_inclusion_param_set_id': incl_set_id}
        keys.append((spont_timescales.TwopTauInclusion & primary_key & 'is_good_tau_roi=1').fetch('KEY'))
    keys    = [entries for subkeys in keys for entries in subkeys] # flatten
    
    # retrieve taus 
    taus    = np.array((spont_timescales.TwopTau & keys).fetch('tau'))

    # figure out how many of those are somas
    seg_keys   = VM['twophoton'].Segmentation2P & keys
    is_soma    = np.array((VM['twophoton'].Roi2P & seg_keys).fetch('is_soma'))
    total_soma = np.sum(is_soma)
    
    end_time = time.time()
    print("     done after {: 1.1f} min".format((end_time-start_time)/60))
    
    return taus, keys, total_soma

# ---------------
# %% retrieve all x-corr taus for a given area and set of parameters, from dj database
def get_all_tau_xcorr(area, params = params, dff_type = 'residuals_dff'):
    
    start_time      = time.time()
    print('Fetching all x-corr taus for {}...'.format(area))
        
    # get primary keys for query
    mice              = params['general_params']['{}_mice'.format(area)]
    corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
    tau_param_set_id  = params['general_params']['tau_param_set_id']
    incl_set_id       = params['general_params']['twop_inclusion_param_set_id']
    
    # get relavant keys 
    keys = list()
    for mouse in mice:
        primary_key = {'subject_fullname': mouse, 'corr_param_set_id': corr_param_set_id, 'tau_param_set_id': tau_param_set_id, 'twop_inclusion_param_set_id': incl_set_id}
        keys.append((spont_timescales.TwopXcorrTauInclusion & primary_key & 'is_good_xcorr_tau=1').fetch('KEY'))
    keys    = [entries for subkeys in keys for entries in subkeys] # flatten
    
    # retrieve taus 
    taus    = np.array((spont_timescales.TwopXcorrTau & keys).fetch('tau'))

    end_time = time.time()
    print("     done after {: 1.1f} min".format((end_time-start_time)/60))
    
    return taus, keys

# ---------------
# %% statistically compare taus across areas and plot
def plot_area_tau_comp(params=params, dff_type='residuals_dff', axis_handle=None, v1_taus=None, m2_taus=None, corr_type='autocorr'):

    # get taus
    if corr_type == 'autocorr':
        if v1_taus is None:
            v1_taus, _ , _ = get_all_tau('V1',params=params,dff_type=dff_type)
        if m2_taus is None:
            m2_taus, _ , _ = get_all_tau('M2',params=params,dff_type=dff_type)
        histbins = params['tau_hist_bins']
        n_is     = 'neurons'
        
    elif corr_type == 'xcorr':
        if v1_taus is None:
            v1_taus, _ = get_all_tau_xcorr('V1',params=params,dff_type=dff_type)
        if m2_taus is None:
            m2_taus, _ = get_all_tau_xcorr('M2',params=params,dff_type=dff_type)
        histbins = params['tau_hist_bins_xcorr']
        n_is     = 'pairs'
       
    elif corr_type == 'eigen':
        if v1_taus is None:
            v1_taus, _ = get_rec_xcorr_eigen_taus('V1',params=params,dff_type=dff_type)
        if m2_taus is None:
            m2_taus, _ = get_rec_xcorr_eigen_taus('M2',params=params,dff_type=dff_type)
        histbins = params['tau_hist_bins_eigen']
        n_is     = 'fovs'
         
    else:    
        print('unknown corr_type, doing nothing')
        return
    
    # compute stats
    tau_stats = dict()
    tau_stats['analysis_params'] = deepcopy(params)
    tau_stats['V1_num_'+ n_is]   = np.size(v1_taus)
    tau_stats['V1_mean']         = np.mean(v1_taus)
    tau_stats['V1_sem']          = np.std(v1_taus,ddof=1) / np.sqrt(tau_stats['V1_num_'+ n_is]-1)
    tau_stats['V1_median']       = np.median(v1_taus)
    tau_stats['V1_iqr']          = scipy.stats.iqr(v1_taus)
    tau_stats['M2_num_'+ n_is]   = np.size(m2_taus)
    tau_stats['M2_mean']         = np.mean(m2_taus)
    tau_stats['M2_sem']          = np.std(m2_taus,ddof=1) / np.sqrt(tau_stats['M2_num_'+ n_is]-1)
    tau_stats['M2_median']       = np.median(m2_taus)
    tau_stats['M2_iqr']          = scipy.stats.iqr(m2_taus)
    tau_stats['pval'], tau_stats['test_name'] = general_stats.two_group_comparison(v1_taus, m2_taus, is_paired=False, tail="two-sided")

    # plot
    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle

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
    ax.set_ylabel('Prop. ' + n_is)        
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return tau_stats, ax 

# ---------------
# %% get centroid and sess info for a list of tau keys
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

# ---------------
# %% statistically compare taus across areas and plot
def clustering_by_tau(taus, centroids, rec_ids, params=params, rng=None):

    # set random seed and delete low /  high tau values if applicable
    start_time      = time.time()
    print('Performing clustering analysis...')
    if rng is None:
        rng = np.random.default_rng(seed=params['random_seed'])
    
    if params['max_tau'] is not None:
        del_idx   = np.argwhere(taus>params['max_tau'])
        taus      = np.delete(taus,del_idx)
        centroids = np.delete(centroids,del_idx,axis=0)
        rec_ids   = np.delete(rec_ids,del_idx)
    if params['min_tau'] is not None:
        del_idx   = np.argwhere(taus<params['min_tau'])
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
    clust_results['analysis_params']      = deepcopy(params)
    clust_results['dist_um']              = bins[:-1]+np.diff(bins)[0]/2
    clust_results['num_boot_iter']        = num_iter  
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

# ---------------
# %% statistically compare taus across areas and plot
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

# ---------------
# %% plot taus on FOV
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

# ---------------
# %% retrieve all x-corr taus for a given area and set of parameters using matrix eigenvalues
def get_rec_xcorr_eigen_taus(area, params = params, dff_type = 'residuals_dff'):
    
    start_time      = time.time()
    print('Fetching all xcorr mats for {}...'.format(area))
        
    # get primary keys for query
    sess_keys = get_single_sess_keys(area, params=params, dff_type='residuals_dff')
    
    # get relavant keys, filtering for inclusion for speed
    xcorr_mats = list()
    taus       = list()
    for sess in sess_keys:
        # get keys for good pairs in this session
        sess['corr_param_set_id'] = params['general_params']['corr_param_id_{}'.format(dff_type)]
        sess['tau_param_set_id']  = params['general_params']['tau_param_set_id']
        sess['twop_inclusion_param_set_id'] = params['general_params']['twop_inclusion_param_set_id']
        good_keys = (spont_timescales.TwopXcorrInclusion & sess & 'is_good_xcorr_pair=1').fetch('KEY')
        
        # figure out total number of neurons
        seg_key   = twop_opto_analysis.get_single_segmentation_key(sess)
        is_neuron = np.array((VM['twophoton'].Roi2P & seg_key).fetch('roi_type'))=='soma'
        f_period  = 1/((VM['twophoton'].Scan & seg_key).fetch1('frame_rate_hz'))
        
        # compute mat
        this_mat = xcorr_mat_from_keys(good_keys,is_neuron)
        xcorr_mats.append(this_mat)
        taus.append(tau_from_eigenval(this_mat,f_period))
        
    end_time = time.time()
    print("     done after {: 1.1f} min".format((end_time-start_time)/60))
    
    return np.array(taus), xcorr_mats

# ---------------
# %% build a symmetrical x-corr matrix from dj keys
def xcorr_mat_from_keys(xcorr_keys, is_neuron):
        
    neuron_idx  = np.argwhere(is_neuron==1)
    num_neurons = np.size(neuron_idx)    
    xcorr_mat   = np.zeros((num_neurons,num_neurons))
    for key in xcorr_keys:
        idx1 = np.argwhere(neuron_idx==key['roi_id_1']-1)
        idx2 = np.argwhere(neuron_idx==key['roi_id_2']-1)
        xcorr_vals = (spont_timescales.TwopXcorr & key).fetch1('xcorr_vals')
        max_val    = np.max(xcorr_vals)
        min_val    = np.min(xcorr_vals)
        if max_val >= np.abs(min_val):
            val = max_val
        else:
            val = min_val
        xcorr_mat[idx1,idx2] = val
        xcorr_mat[idx2,idx1] = val

    return xcorr_mat

# ---------------
# %% timescale from eigenvalue of the x-corr matrix
def tau_from_eigenval(xcorr_mat,frame_period):
  
    # tau will be defined by the longest timescale, 
    # given by the reciprocal of the smallest-magnitude negative eigenvalue
    eigvals = np.real(np.linalg.eigvals(xcorr_mat))  
    eigvals = eigvals[eigvals<0]
    tau     = 1/np.abs(np.max(eigvals))
    
    return tau*frame_period