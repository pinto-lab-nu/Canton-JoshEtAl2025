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
from utils.plotting import plot_fov_heatmap_blur
import time

# connect to dj 
VM     = connect_to_dj.get_virtual_modules()

# %% declare default params 
params = {
        'random_seed'               : 42, 
        'max_tau'                   : None,
        'min_tau'                   : 0.1,
        'tau_hist_bins'             : np.arange(0,10.2,.2),
        'tau_hist_bins_xcorr'       : np.arange(0,20.2,.2),
        'tau_hist_bins_eigen'       : np.arange(0,101,1),
        'clustering_dist_bins'      : np.arange(0,350,50),
        'clustering_num_boot_iter'  : 10000,
        'clustering_num_shuffles'   : 1000,
        'clustering_zscore_taus'    : True,
        }
params['general_params'] = {
        # 'V1_mice'     : ['jec822_NCCR62','jec822_NCCR63','jec822_NCCR66','jec822_NCCR86'] ,
        'V1_mice'     : ['jec822_NCCR63','jec822_NCCR66','jec822_NCCR86','jec822_NCCR121','jec822_NCCR150'],
        'M2_mice'     : ['jec822_NCCR32','jec822_NCCR72','jec822_NCCR73','jec822_NCCR77','jec822_NCCR80','jec822_NCCR141'] ,
        # 'V1_mice'     : ['jec822_NCCR121'] ,
        # 'V1_mice'     : ['jec822_NCCR63','jec822_NCCR66','jec822_NCCR86','jec822_NCCR121','jec822_NCCR150'],
        # 'M2_mice'     : ['jec822_NCCR80'] ,
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
        'tau_param_set_id'               : 2,
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
        
    # if in session keys delete, later
    return sess_keys
# %%
def get_twop_tau_keys(area, params, dff_type, scan_number=1,session_number=1):
   """
   Returns all relevant keys from TwopTauInclusion for a given area, data type, and scan number.

   Parameters:
   - area: str, like "V1" or "M2"
   - params: dict, containing the necessary 'general_params'
   - dff_type: str, e.g. 'residuals_dff'
   - scan_number: int, default is 1

   Returns:
   - keys: list of DataJoint primary keys (dicts)
   """
   mice = params['general_params'][f'{area}_mice']
   corr_param_set_id = params['general_params'][f'corr_param_id_{dff_type}']
   tau_param_set_id = params['general_params']['tau_param_set_id']
   incl_set_id = params['general_params']['twop_inclusion_param_set_id']

   keys = []
   for mouse in mice:
       primary_key = {
           'subject_fullname': mouse,
           'corr_param_set_id': corr_param_set_id,
           'tau_param_set_id': tau_param_set_id,
           'twop_inclusion_param_set_id': incl_set_id,
            'scan_number': scan_number,
            'session_number':session_number
       }
       # Fetch and extend list
       keys.extend((spont_timescales.TwopTauInclusion & primary_key).fetch('KEY'))

   return keys

# %%  Takes a single cell key and gives the fit data, useful for plotting individual examples of fit
def extract_fit_data_for_key(key):
    """
    Given a DataJoint key, fetch and organize all necessary parameters
    for plotting autocorrelation with mono and dual exponential fits.
    
    Returns:
    - acorr_vector: 1D numpy array
    - time_vector: 1D numpy array
    - mono_fit_params: dict
    - dual_fit_params: dict
    - r2_s: float (R² for mono fit)
    - bic_s: float (BIC for mono fit)
    - r2_d: float (R² for dual fit)
    - bic_d: float (BIC for dual fit)
    """
    # Fetch data from tables
    acorr, lags = (spont_timescales.TwopAutocorr & key).fetch1('autocorr_vals', 'lags_sec')
    
    tau_value = (spont_timescales.TwopTau & key).fetch('tau')[0]
    if tau_value <= 0:
        raise ValueError(f"Tau must be > 0, got {tau_value}")

    r2_s = (spont_timescales.TwopTau & key).fetch('r2_fit_single')[0]
    r2_d = (spont_timescales.TwopTau & key).fetch('r2_fit_double')[0]

    bic_s = (spont_timescales.TwopTau & key).fetch('bic_single')[0]
    bic_d = (spont_timescales.TwopTau & key).fetch('bic_double')[0]

    mono_fit = (spont_timescales.TwopTau & key).fetch('single_fit_params')[0]
    dual_fit = (spont_timescales.TwopTau & key).fetch('double_fit_params')[0]

    # Build time vector to match acorr
    time_vector = lags[:len(acorr)]  # ensure matched length

    # Reformat params
    mono_fit_params = {
        'A': mono_fit['A_fit_mono'],
        'tau': mono_fit['tau_fit_mono'],
        'offset': mono_fit['offset_fit_mono']
    }

    dual_fit_params = {
        'A0': dual_fit['A_fit_0'],
        'tau0': dual_fit['tau_fit_0'],
        'A1': dual_fit['A_fit_1'],
        'tau1': dual_fit['tau_fit_1'],
        'offset': dual_fit['offset_fit_dual']
    }

    return (
        acorr,
        time_vector,
        mono_fit_params,
        dual_fit_params,
        r2_s,
        bic_s,
        r2_d,
        bic_d
    )




# ---------------
# %% retrieve all taus for a given area and set of parameters, from dj database
# def get_all_tau(area, params = params, dff_type = 'residuals_dff'):
    
#     """
#     get_all_tau(area, params=params, dff_type='residuals_dff')
#     fetches timescales for every neuron in a given area
    
#     INPUT:
#     area: 'V1' or 'M2'
#     params: dictionary as the one on top of this file, used to determine
#             mice to look for and which parameter sets correspond to desired
#             dff_type
#     dff_type: 'residuals_dff' for residuals of running linear regression (default)
#               'residuals_deconv' for residuals of running Poisson GLM on deconvolved traces
#               'noGlm_dff' for plain dff traces
              
#     OUTPUT:
#     taus: vector with taus that pass inclusion criteria 
#           in params['general_params']['twop_inclusion_param_set_id']
#     keys: list of keys corresponding to each tau
#     total_soma: total number of somas regardless of inclusion criteria
#     """
    
#     start_time      = time.time()
#     print('Fetching all taus for {}...'.format(area))
        
#     # get primary keys for query
#     mice              = params['general_params']['{}_mice'.format(area)]
#     corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
#     tau_param_set_id  = params['general_params']['tau_param_set_id']
#     incl_set_id       = params['general_params']['twop_inclusion_param_set_id']
    
#     # get relavant keys, filtering for inclusion for speed
#     keys = list()
#     for mouse in mice:
#         primary_key = {'subject_fullname': mouse, 'corr_param_set_id': corr_param_set_id, 'tau_param_set_id': tau_param_set_id, 'twop_inclusion_param_set_id': incl_set_id}
#         keys.append((spont_timescales.TwopTauInclusion & primary_key & 'is_good_tau_roi=1').fetch('KEY'))
#     keys    = [entries for subkeys in keys for entries in subkeys] # flatten
    
#     # retrieve taus 
#     taus    = np.array((spont_timescales.TwopTau & keys).fetch('tau'))
#     taus = taus[taus > 0.1]

#     # figure out how many of those are somas
#     seg_keys   = VM['twophoton'].Segmentation2P & keys
#     is_soma    = np.array((VM['twophoton'].Roi2P & seg_keys).fetch('is_soma'))
#     total_soma = np.sum(is_soma)
    
#     end_time = time.time()
#     print("     done after {: 1.1f} min".format((end_time-start_time)/60))
    
#     return taus, keys, total_soma
# %% retrieve all taus for a given area and set of parameters, from dj database
# added cutoff for tau

def get_all_tau(area, params=params, dff_type='residuals_dff'):
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

    start_time = time.time()
    print('Fetching all taus for {}...'.format(area))

    # get primary keys for query
    mice = params['general_params']['{}_mice'.format(area)]
    corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
    tau_param_set_id = params['general_params']['tau_param_set_id']
    incl_set_id = params['general_params']['twop_inclusion_param_set_id']

    # get relevant keys, filtering for inclusion for speed
    keys = list()
    for mouse in mice:
        primary_key = {
            'subject_fullname': mouse,
            'corr_param_set_id': corr_param_set_id,
            'tau_param_set_id': tau_param_set_id,
            'twop_inclusion_param_set_id': incl_set_id
        }
        keys.append((spont_timescales.TwopTauInclusion & primary_key & 'is_good_tau_roi=1').fetch('KEY'))
    keys = [entry for sublist in keys for entry in sublist]  # flatten

    # fetch taus and associated keys, then filter for tau > 0.1
    tau_entries = (spont_timescales.TwopTau & keys).fetch('tau', 'KEY')
    taus_all, keys_all = tau_entries
    mask = taus_all > 0.1
    taus = taus_all[mask]
    keys = [keys_all[i] for i in np.where(mask)[0]]

    # figure out how many of those are somas
    seg_keys = VM['twophoton'].Segmentation2P & keys
    is_soma = np.array((VM['twophoton'].Roi2P & seg_keys).fetch('is_soma'))
    total_soma = np.sum(is_soma)

    end_time = time.time()
    print("     done after {: 1.1f} min".format((end_time - start_time) / 60))

    return taus, keys, total_soma

# ---------------
# %% retrieve all taus for a full list of roi keys, from dj database
def get_tau_from_roi_keys(roi_keys, params = params, dff_type = 'residuals_dff', verbose=True):
    
    """
    get_tau_from_roi_keys(roi_keys, params=params, dff_type='residuals_dff')
    fetches timescales for every neuron for a list of roi keys
    
    INPUT:
    roi_keys: list of roi segmentation keys
    params: dictionary as the one on top of this file, used to determine
            mice to look for and which parameter sets correspond to desired
            dff_type
    dff_type: 'residuals_dff' for residuals of running linear regression (default)
              'residuals_deconv' for residuals of running Poisson GLM on deconvolved traces
              'noGlm_dff' for plain dff traces
    verbose: whether to print progress (default is True)
              
    OUTPUT:
    taus_dict: dictionary with taus and inclusion for each roi
    """
    
    if verbose:
        start_time      = time.time()
        print('Fetching all taus from roi keys...')
        
    # get primary keys for query
    corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
    tau_param_set_id  = params['general_params']['tau_param_set_id']
    incl_set_id       = params['general_params']['twop_inclusion_param_set_id']
    param_key         = {'corr_param_set_id': corr_param_set_id, 'tau_param_set_id': tau_param_set_id, 'twop_inclusion_param_set_id': incl_set_id}
 
    # retrieve taus and inclusion
    taus    = np.array((spont_timescales.TwopTau & roi_keys & param_key).fetch('tau'))
    is_good = np.array((spont_timescales.TwopTauInclusion & roi_keys & param_key).fetch('is_good_tau_roi'))
    

    taus_dict = {
                'roi_keys'    : roi_keys,
                'taus'        : taus,
                'is_good_tau' : is_good,
                'params'      : deepcopy(params),
                'dff_type'    : dff_type
                  }
    
    if verbose:
        end_time = time.time()
        print("     done after {: 1.1f} min".format((end_time-start_time)/60))
    
    return taus_dict


# # %%
# def get_tau_from_roi_keys(roi_keys, params=params, dff_type='residuals_dff', verbose=True):
#     """
#     get_tau_from_roi_keys(roi_keys, params=params, dff_type='residuals_dff')
#     fetches timescales for every neuron for a list of roi keys

#     INPUT:
#     roi_keys: list of roi segmentation keys
#     params: dictionary as the one on top of this file, used to determine
#             mice to look for and which parameter sets correspond to desired
#             dff_type
#     dff_type: 'residuals_dff' for residuals of running linear regression (default)
#               'residuals_deconv' for residuals of running Poisson GLM on deconvolved traces
#               'noGlm_dff' for plain dff traces
#     verbose: whether to print progress (default is True)

#     OUTPUT:
#     taus_dict: dictionary with taus and inclusion for each roi
#     """

#     if verbose:
#         start_time = time.time()
#         print('Fetching all taus from roi keys...')

#     # get primary keys for query
#     corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
#     tau_param_set_id = params['general_params']['tau_param_set_id']
#     incl_set_id = params['general_params']['twop_inclusion_param_set_id']
#     param_key = {
#         'corr_param_set_id': corr_param_set_id,
#         'tau_param_set_id': tau_param_set_id,
#         'twop_inclusion_param_set_id': incl_set_id
#     }

#     # retrieve taus and associated keys
#     tau_entries = (spont_timescales.TwopTau & roi_keys & param_key).fetch('tau', 'KEY')
#     taus_all, keys_all = tau_entries
#     mask = taus_all > 0.1
#     taus = taus_all[mask]
#     filtered_keys = [keys_all[i] for i in np.where(mask)[0]]

#     # retrieve inclusion only for filtered keys
#     is_good = np.array((spont_timescales.TwopTauInclusion & filtered_keys & param_key).fetch('is_good_tau_roi'))

#     taus_dict = {
#         'roi_keys': filtered_keys,
#         'taus': taus,
#         'is_good_tau': is_good,
#         'params': deepcopy(params),
#         'dff_type': dff_type
#     }

#     if verbose:
#         end_time = time.time()
#         print("     done after {: 1.1f} min".format((end_time - start_time) / 60))

#     return taus_dict





# ---------------
# %% retrieve all x-corr taus for a given area and set of parameters, from dj database
def get_all_tau_xcorr(area, params = params, dff_type = 'residuals_dff'):
    
    """
    get_all_tau_xcorr(area, params=params, dff_type='residuals_dff')
    fetches timescales of x-corr for every neuron pair in a given area
    
    INPUT:
    area: 'V1' or 'M2'
    params: dictionary as the one on top of this file
    
    OUTPUT:
    taus: vector with taus that pass inclusion criteria 
          in params['general_params']['twop_inclusion_param_set_id']
    keys: list of dj keys corresponding to each tau
    """
    
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
def plot_area_tau_comp(params, dff_type='residuals_dff', axis_handle=None,
                       v1_taus=None, m2_taus=None, corr_type='autocorr',
                       log_x=True, xlim=None, ylim=None):
    """
    plot_area_tau_comp(params, dff_type='residuals_dff', axis_handle=None,
                       v1_taus=None, m2_taus=None, corr_type='autocorr',
                       log_x=True, xlim=None, ylim=None)
    
    Compares taus across areas and plots them.

    INPUT:
    - params: dictionary with analysis parameters
    - dff_type: signal type to analyze
    - axis_handle: optional matplotlib axis to plot on
    - v1_taus, m2_taus: optional vectors of taus; if None, will be fetched
    - corr_type: 'autocorr', 'xcorr', or 'eigen'
    - log_x: if True (default), sets x-axis to log scale
    - xlim: optional tuple for x-axis limits (e.g., (0, 5))
    - ylim: optional tuple for y-axis limits (e.g., (0, 1))

    OUTPUT:
    - tau_stats: dictionary with statistics
    - ax: axis handle
    """

    # Get taus based on correlation type
    if corr_type == 'autocorr':
        if v1_taus is None:
            v1_taus, _, _ = get_all_tau('V1', params=params, dff_type=dff_type)
        if m2_taus is None:
            m2_taus, _, _ = get_all_tau('M2', params=params, dff_type=dff_type)
        histbins = params['tau_hist_bins']
        n_is = 'neurons'

    elif corr_type == 'xcorr':
        if v1_taus is None:
            v1_taus, _ = get_all_tau_xcorr('V1', params=params, dff_type=dff_type)
        if m2_taus is None:
            m2_taus, _ = get_all_tau_xcorr('M2', params=params, dff_type=dff_type)
        histbins = params['tau_hist_bins_xcorr']
        n_is = 'pairs'

    elif corr_type == 'eigen':
        if v1_taus is None:
            v1_taus, _ = get_rec_xcorr_eigen_taus('V1', params=params, dff_type=dff_type)
        if m2_taus is None:
            m2_taus, _ = get_rec_xcorr_eigen_taus('M2', params=params, dff_type=dff_type)
        histbins = params['tau_hist_bins_eigen']
        n_is = 'fovs'

    else:
        print('Unknown corr_type, doing nothing.')
        return

    # Compute stats
    tau_stats = dict()
    tau_stats['analysis_params'] = deepcopy(params)
    tau_stats['V1_num_' + n_is] = np.size(v1_taus)
    tau_stats['V1_mean'] = np.mean(v1_taus)
    tau_stats['V1_sem'] = np.std(v1_taus, ddof=1) / np.sqrt(tau_stats['V1_num_' + n_is] - 1)
    tau_stats['V1_median'] = np.median(v1_taus)
    tau_stats['V1_iqr'] = scipy.stats.iqr(v1_taus)
    tau_stats['M2_num_' + n_is] = np.size(m2_taus)
    tau_stats['M2_mean'] = np.mean(m2_taus)
    tau_stats['M2_sem'] = np.std(m2_taus, ddof=1) / np.sqrt(tau_stats['M2_num_' + n_is] - 1)
    tau_stats['M2_median'] = np.median(m2_taus)
    tau_stats['M2_iqr'] = scipy.stats.iqr(m2_taus)
    tau_stats['pval'], tau_stats['test_name'] = general_stats.two_group_comparison(
        v1_taus, m2_taus, is_paired=False, tail="two-sided")

    # Create plot
    if axis_handle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axis_handle

    v1_counts, _ = np.histogram(v1_taus, bins=histbins, density=False)
    m2_counts, _ = np.histogram(m2_taus, bins=histbins, density=False)
    xaxis = histbins[:-1] + np.diff(histbins)[0]

    ax.plot(xaxis, np.cumsum(v1_counts) / np.sum(v1_counts),
            color=params['general_params']['V1_cl'],
            label=params['general_params']['V1_lbl'])
    ax.plot(xaxis, np.cumsum(m2_counts) / np.sum(m2_counts),
            color=params['general_params']['M2_cl'],
            label=params['general_params']['M2_lbl'])

    # Plot median markers and annotate them
    ax.plot(tau_stats['V1_median'], 0.03, 'v', color=params['general_params']['V1_cl'])
    ax.plot(tau_stats['M2_median'], 0.03, 'v', color=params['general_params']['M2_cl'])
    ax.text(tau_stats['V1_median'], 0.05, f"Med: {tau_stats['V1_median']:.2f}",
            color=params['general_params']['V1_cl'], ha='center', fontsize=8)
    ax.text(tau_stats['M2_median'], 0.07, f"Med: {tau_stats['M2_median']:.2f}",
            color=params['general_params']['M2_cl'], ha='center', fontsize=8)

    ax.text(0.95, 0.9, f'p = {tau_stats["pval"]:.2g}', transform=ax.transAxes)

    # Axis scale and limits
    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel('$\\tau$ (sec)')
    ax.set_ylabel('Prop. ' + n_is)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return tau_stats, ax

# ---------------
# %% get centroid and sess info for a list of tau keys
def get_centroids_by_rec(tau_keys):

    """
    get_centroids_by_rec(tau_keys)
    returns centroids and session ids for a list of tau keys
    
    INPUT:
    tau_keys: list of dj keys for tau table
    
    OUTPUT:
    centroids: numpy array with [x,y] centroids of each roi in um
    sess_ids : array with arbitrary session id integers
    """
    
    # find unique sessions 
    sess_ids = sess_ids_from_tau_keys(tau_keys)

    # get centroids and assign numeric session ids
    centroids = list()
    for this_key in tau_keys:
        centr    = (VM['twophoton'].Roi2P & this_key).fetch1('centroid_pxls')
        xs, ys   = (VM['twophoton'].Scan & this_key).fetch1('microns_per_pxl_x','microns_per_pxl_y')
        centroids.append(np.array([centr[0,0]*ys, centr[0,1]*xs]))
    
    return np.array(centroids), np.array(sess_ids) 

# ---------------
# %% get centroid and sess info for a list of tau keys
def sess_ids_from_tau_keys(tau_keys):

    """
    sess_ids_from_tau_keys(tau_keys)
    returns session ids for a list of tau keys
    
    INPUT:
    tau_keys: list of dj keys for tau table (or other tables with roi info)
    
    OUTPUT:
    sess_ids : array with arbitrary session id integers
    """
    
    # find unique sessions 
    subj_date_list = ['{}-{}'.format(this_key['subject_fullname'],this_key['session_date']) for this_key in tau_keys] 
    unique_sess    = np.unique(subj_date_list).tolist()
    sess_ids       = list()
    sess_ids       = list()
    for sess in subj_date_list:
        sess_ids.append(unique_sess.index(sess))
    
    return np.array(sess_ids) 

# ---------------
# %% spatial clustering analysis by tau differences
def clustering_by_tau(taus, centroids, rec_ids, params=params, rng=None):

    """
    clustering_by_tau(taus, centroids, rec_ids, params=params, rng=None)
    performs spatial clustering analysis by tau differences 
    
    INPUT:
    taus: vector of tau values
    centroids: numpy array with [x,y] centroids of each roi in um
    rec_ids: vector of session ids for each roi
    params: dictionary as the one on top of this file
    rng: random number generator
    
    OUTPUT:
    clust_results: dictionary with clustering results
    tau_mat: matrix with tau differences for each bootstrapped iteration
    """
    
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
# %% statistically compare tau clustering across areas and plot
def plot_clustering_comp(v1_clust=None, m2_clust=None, params=params, axis_handle=None):
    """
    plot_clustering_comp(v1_clust=None, m2_clust=None, params=params, axis_handle=None)
    compares clustering results across areas and plots them

    INPUT:
    v1_clust: dictionary with clustering results for V1
    m2_clust: dictionary with clustering results for M2
    params: parameter dictionary
    axis_handle: optional axis handle to plot on

    OUTPUT:
    ax: axis handle of plot
    """

    import matplotlib.pyplot as plt
    import numpy as np

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

    # Plot error bars
    ax.errorbar(x=xaxis, y=v1_mean, yerr=v1_std,
                color=params['general_params']['V1_cl'],
                label=params['general_params']['V1_lbl'],
                marker='.')
    ax.errorbar(x=xaxis, y=m2_mean, yerr=m2_std,
                color=params['general_params']['M2_cl'],
                label=params['general_params']['M2_lbl'],
                marker='.')

    # Zero line
    ax.plot([xaxis[0], xaxis[-1]], [0, 0], '--', color='gray')

    # Plot asterisks for significant p-values
    for x, y, sig in zip(xaxis, v1_mean - v1_std - 0.01, v1_clust['tau_diff_bydist_isSig']):
        if sig:
            ax.text(x, y, '*', color=params['general_params']['V1_cl'],
                    ha='center', va='top', fontsize=12)

    for x, y, sig in zip(xaxis, m2_mean - m2_std - 0.015, m2_clust['tau_diff_bydist_isSig']):
        if sig:
            ax.text(x, y, '*', color=params['general_params']['M2_cl'],
                    ha='center', va='top', fontsize=12)

    # Labels and legend
    ylabel = '|$\\tau$ diff (z-score)| - shuffle' if params['clustering_zscore_taus'] else '|$\\tau$ diff (sec)| - shuffle'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Pairwise dist. $\\mu$m')
    ax.legend()

    # Aesthetics
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax

# ---------------
# %% plot taus on FOV
def plot_tau_fov(tau_keys, sess_ids, which_sess=0, do_zscore=params['clustering_zscore_taus'], prctile_cap=[0,95], axis_handle=None, fig_handle=None,max_min=None,cmap=None):

    """
    plot_tau_fov(tau_keys, sess_ids, which_sess=0, do_zscore=params['clustering_zscore_taus'], prctile_cap=[0,95], axis_handle=None, fig_handle=None)
    plots taus as a heatmap on FOV
    
    INPUT:
    tau_keys: list of dj keys for tau table
    sess_ids: vector of session ids for each roi
    which_sess: session id to plot (default is 0)
    do_zscore: whether to z-score taus (default is params['clustering_zscore_taus'])
    prctile_cap: bottom and top percentile cap for color scale (default is [0,95])
    axis_handle: optional axis handle to plot on
    fig_handle: optional figure handle to plot on
    
    OUTPUT:
    ax: axis handle of plot
    fig: figure handle of plot
    """
    if cmap==None:
        cmap='rocket'
    
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
    ax, fig = plot_fov_heatmap_blur(roi_vals=taus, roi_coords=roi_coords, im_size=im_size, um_per_pxl=um_per_pxl, \
                              prctile_cap=prctile_cap, cbar_lbl=lbl, axisHandle=axis_handle, figHandle=fig_handle,plot_colorbar=True,
                              max_min=max_min,cmap=cmap)
        
   
    
    return ax, fig

# ---------------
# %% retrieve all x-corr taus for a given area and set of parameters using matrix eigenvalues
def get_rec_xcorr_eigen_taus(area, params = params, dff_type = 'residuals_dff'):
    
    """
    get_rec_xcorr_eigen_taus(area, params=params, dff_type='residuals_dff')
    estimates timescales of an fov using eigenvalues of the symmetrical max(x-corr) matrix
    
    INPUT:
    area: 'V1' or 'M2'
    params: dictionary as the one on top of this file
    dff_type: 'residuals_dff' for residuals of running linear regression (default)
              'residuals_deconv' for residuals of running Poisson GLM on deconvolved traces
              'noGlm_dff' for plain dff traces
              
    OUTPUT:
    taus: vector with single tau per session 
    xcorr_mats: list of x-corr matrices
    """
    
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
    """
    xcorr_mat_from_keys(xcorr_keys, is_neuron)
    builds a symmetrical x-corr matrix from dj keys, called by get_rec_xcorr_eigen_taus
    
    INPUT:
    xcorr_keys: list of dj keys for x-corr table
    is_neuron: boolean vector with whether each roi is a neuron
    
    OUTPUT:
    xcorr_mat: symmetrical x-corr matrix
    """
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
  
    """
    tau_from_eigenval(xcorr_mat,frame_period)
    estimates timescale from the eigenvalues of the x-corr matrix
    called by get_rec_xcorr_eigen_taus
    
    INPUT:
    xcorr_mat: symmetrical x-corr matrix
    frame_period: frame period in seconds
    
    OUTPUT:
    tau: estimated timescale
    """
    
    # tau will be defined by the longest timescale, 
    # given by the reciprocal of the smallest-magnitude negative eigenvalue
    eigvals = np.real(np.linalg.eigvals(xcorr_mat))  
    eigvals = eigvals[eigvals<0]
    tau     = 1/np.abs(np.max(eigvals))
    
    return tau*frame_period

# %%

from scipy.optimize import curve_fit
import numpy as np

def single_exp(x, a, tau, c):
    return a * np.exp(-x / tau) + c

def double_exp(x, a1, tau1, a2, tau2, c):
    return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + c

def compute_r2(y_true, y_pred):
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

def compute_bic(y_true, y_pred, num_params):
    n = len(y_true)
    residual = y_true - y_pred
    sse = np.nansum(residual ** 2)
    if sse <= 0 or n <= 0:
        return np.nan
    bic = n * np.log(sse / n) + num_params * np.log(n)
    return bic

def fit_exponentials_from_df(
    acorr_df,
    exclude_low_acorr=False,
    acorr_threshold=0.1,
    max_lag=None,
    bounded_fit=False,
    tau_bounds=(0.1, 100)
):
    """
    Fit single and double exponential models to autocorrelation data in a DataFrame.

    Parameters:
    - acorr_df: DataFrame where each row is an autocorrelation trace
    - exclude_low_acorr: bool, whether to skip traces with weak acorr[1] values
    - acorr_threshold: float, threshold for acorr[1] to include trace
    - max_lag: int, optional max number of lags to fit
    - bounded_fit: bool, whether to apply bounds to exponential fits
    - tau_bounds: tuple (min_tau, max_tau), bounds on tau parameter for both fits

    Returns:
    - results: list of dictionaries with fit results
    - filtered_acorr: list of autocorrelation arrays used in fitting
    - valid_indices: list of row indices in acorr_df that were successfully fit
    """
    from scipy.optimize import curve_fit

    results = []
    filtered_acorr = []
    valid_indices = []

    tau_min, tau_max = tau_bounds

    for i, row in acorr_df.iterrows():
        y_full = row.to_numpy()
        x_full = np.arange(len(y_full))

        # Skip if invalid
        if len(y_full) == 0 or np.all(np.isnan(y_full)) or np.all(y_full == 0) or np.all(y_full == y_full[0]):
            continue

        acorr1_val = y_full[1] - np.nanmin(y_full) if len(y_full) > 1 else np.nan
        if exclude_low_acorr and acorr1_val < acorr_threshold:
            continue

        filtered_acorr.append(y_full)
        valid_indices.append(i)

        for trim in [False, True]:
            x = x_full[2:] if trim else x_full
            y = y_full[2:] if trim else y_full

            if max_lag is not None:
                x = x[:max_lag]
                y = y[:max_lag]

            if len(y) < 5:
                continue

            entry = {
                'index': i,
                'exclude_first_two': trim,
                'acorr1': acorr1_val
            }

            # --- Single exponential fit ---
            try:
                if bounded_fit:
                    bounds_s = ([0, tau_min, -np.inf], [np.inf, tau_max, np.inf])
                    popt_s, _ = curve_fit(single_exp, x, y, p0=(1, 1, 0), bounds=bounds_s, maxfev=10000)
                else:
                    popt_s, _ = curve_fit(single_exp, x, y, p0=(1, 1, 0), maxfev=10000)

                y_pred_s = single_exp(x, *popt_s)
                r2_s = compute_r2(y, y_pred_s)
                bic_s = compute_bic(y, y_pred_s, num_params=3)
                entry_single = {
                    **entry,
                    'type': 'single_exp',
                    'params': dict(zip(['a', 'tau', 'c'], popt_s)),
                    'r2': r2_s,
                    'bic': bic_s
                }
            except Exception:
                entry_single = {
                    **entry,
                    'type': 'single_exp',
                    'params': {'a': np.nan, 'tau': np.nan, 'c': np.nan},
                    'r2': np.nan,
                    'bic': np.nan
                }

            results.append(entry_single)

            # --- Double exponential fit ---
            try:
                if bounded_fit:
                    bounds_d = (
                        [0, tau_min, 0, tau_min, -np.inf],  # a1, tau1, a2, tau2, c
                        [np.inf, tau_max, np.inf, tau_max, np.inf]
                    )
                    popt_d, _ = curve_fit(
                        double_exp, x, y, p0=(1, 1, 0.5, 5, 0),
                        bounds=bounds_d, maxfev=10000
                    )
                else:
                    popt_d, _ = curve_fit(
                        double_exp, x, y, p0=(1, 1, 0.5, 5, 0), maxfev=10000
                    )

                y_pred_d = double_exp(x, *popt_d)
                r2_d = compute_r2(y, y_pred_d)
                bic_d = compute_bic(y, y_pred_d, num_params=5)
                entry_double = {
                    **entry,
                    'type': 'double_exp',
                    'params': dict(zip(['a1', 'tau1', 'a2', 'tau2', 'c'], popt_d)),
                    'r2': r2_d,
                    'bic': bic_d
                }
            except Exception:
                entry_double = {
                    **entry,
                    'type': 'double_exp',
                    'params': {'a1': np.nan, 'tau1': np.nan, 'a2': np.nan, 'tau2': np.nan, 'c': np.nan},
                    'r2': np.nan,
                    'bic': np.nan
                }

            results.append(entry_double)

    return results, filtered_acorr, valid_indices




# %%

import pandas as pd

def classify_fit_results_simple(fit_results):
    single_inc, single_exc = [], []
    double_inc, double_exc = [], []

    for result in fit_results:
        idx = result['index']
        exclude = result['exclude_first_two']
        r2 = result['r2']
        bic = result['bic']
        acorr1 = result['acorr1']

        if result['type'] == 'single_exp':
            tau = result['params']['tau']
            entry = {'index': idx, 'tau': tau, 'r2': r2, 'bic': bic, 'acorr_1_index': acorr1}
            if exclude:
                single_exc.append(entry)
            else:
                single_inc.append(entry)

        elif result['type'] == 'double_exp':
            tau1 = result['params']['tau1']
            tau2 = result['params']['tau2']
            entry = {'index': idx, 'tau1': tau1, 'tau2': tau2, 'r2': r2, 'bic': bic, 'acorr_1_index': acorr1}
            if exclude:
                double_exc.append(entry)
            else:
                double_inc.append(entry)

    df_single_inc = pd.DataFrame(single_inc)
    df_single_exc = pd.DataFrame(single_exc)
    df_double_inc = pd.DataFrame(double_inc)
    df_double_exc = pd.DataFrame(double_exc)

    avg_r2 = {
        'single_inc': df_single_inc['r2'].mean() if not df_single_inc.empty else float('-inf'),
        'single_exc': df_single_exc['r2'].mean() if not df_single_exc.empty else float('-inf'),
        'double_inc': df_double_inc['r2'].mean() if not df_double_inc.empty else float('-inf'),
        'double_exc': df_double_exc['r2'].mean() if not df_double_exc.empty else float('-inf')
    }
    best_fit_type = max(avg_r2, key=avg_r2.get)

    return df_single_inc, df_single_exc, df_double_inc, df_double_exc, best_fit_type


# %%
def calculate_autocorrelations_df(a_df, signal_range=(0, 10000), max_lags=1000):
    """
    Calculate autocorrelations up to max_lags for each row in a 2D DataFrame.

    Parameters:
    - a_df: DataFrame where each row is a 1D time-series signal
    - signal_range: tuple, start and end indices for slicing the signal
    - max_lags: int, number of lags for autocorrelation

    Returns:
    - acorr_df: DataFrame of shape (n_signals, max_lags + 1)
    """
    acorr_list = []

    for _, row in a_df.iterrows():
        try:
            signal = row.iloc[signal_range[0]:signal_range[1]].to_numpy()
            signal = signal - np.nanmean(signal)

            if np.all(np.isnan(signal)) or len(signal) < max_lags:
                acorr = np.full(max_lags + 1, np.nan)
            else:
                acorr_full = np.correlate(signal, signal, mode='full')
                mid = len(acorr_full) // 2
                acorr = acorr_full[mid:mid + max_lags + 1]
                acorr = acorr / acorr[0] if acorr[0] != 0 else np.full_like(acorr, np.nan)
        except Exception:
            acorr = np.full(max_lags + 1, np.nan)

        acorr_list.append(acorr)

    # Convert to 2D DataFrame
    acorr_df = pd.DataFrame(acorr_list)
    acorr_df.columns = [f"lag_{i}" for i in range(max_lags + 1)]

    return acorr_df

