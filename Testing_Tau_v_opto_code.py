# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:20:42 2025

@author: jec822
"""

opto_data = get_full_resp_stats(area='M2', params=opto_params, expt_type='standard', resp_type='dff', which_neurons='all')

# opto_data =
# %%

sess_ids    = analyzeSpont2P.sess_ids_from_tau_keys(opto_data['roi_keys']) 
stim_ids    = deepcopy(opto_data['stim_ids'])
taus = list()
is_good_tau = list()
r2=list()
acorr_index_1_values=list()
SNR=list()

tau_single_list=list()

for sess in np.unique(sess_ids):
    this_sess = sess_ids==sess
    unique_stim = np.unique(stim_ids[this_sess])
    for stim in unique_stim:
        this_stim = np.logical_and(this_sess,stim_ids==stim)
        keys = list(np.array(opto_data['roi_keys'])[this_stim])
        td  = analyzeSpont2P.get_tau_from_roi_keys(keys, params=opto_params, dff_type='residuals_dff', verbose=False)
        
        
        
        [taus.append(t) for t in td['taus']]
        [is_good_tau.append(ig) for ig in td['is_good_tau']]
        
        r2_fit_double = (spont_timescales.TwopTau & keys).fetch('r2_fit_double', 'KEY')
        # r2.append(r2_fit_double[0])
        r2.extend(r2_fit_double[0])
        
        tau_single=(spont_timescales.TwopTau & keys).fetch('single_fit_params', 'KEY')
        tau_single_list.extend(tau_single[0])
        
        snr=(VM['twophoton'].Snr2P & keys).fetch('snr')
        SNR.extend(snr)

        
        # acorr  = (spont_timescales.TwopAutocorr & keys).fetch('autocorr_vals')
        # index_1_values = [subarray[1] for subarray in acorr]
        # is_good_acorr=np.array([value > 0.1 for value in index_1_values])

        
# is_good_tau = np.where(np.array(taus) > -0.2, 1, is_good_tau)
# is_good_tau = np.where(np.array(taus) > -0.2, 1, is_good_tau)

# is_good_tau = np.where(np.array(taus) < 0.1, 0, is_good_tau)

# is_good_tau = mask1

# a=np.array(r2)

# c= (a> 0.2)

# is_good_tau=c

tau_fit_mono_list = [d['tau_fit_mono'] for d in tau_single_list]

# tau_data = {'taus':np.array(tau_fit_mono_list).flatten(),'is_good_tau':np.array(is_good_tau).flatten()}


tau_data = {'taus':np.array(taus).flatten(),'is_good_tau':np.array(is_good_tau).flatten()}


# %%%

# easy access variables    
is_stimd    = deepcopy(opto_data['is_stimd'])
is_sig      = deepcopy(opto_data['is_sig'])
peak_ts     = deepcopy(opto_data['max_or_min_times_sec'])
peak_mag    = deepcopy(opto_data['max_or_min_vals'])
tau         = deepcopy(tau_data['taus'])
is_good_tau = deepcopy(tau_data['is_good_tau'])
is_good_SNR=np.array([value > 8 for value in SNR])

 
aa=np.logical_and(is_good_tau==1,is_stimd==0)
aaa=np.logical_and(aa,is_sig==1)

 
 # %%
 
 
 
  # select out very long taus if desired
   # if opto_params['tau_vs_opto_max_tau'] is not None:
   #     is_good_tau[tau>params['tau_vs_opto_max_tau']] = 0
     
     
 # do median split on tau
 tau_th      = np.median(tau[is_good_tau==1])
 is_short    = tau < tau_th
 
 # response properties by tau (need to implement peak width)
 # bins     = opto_params['tau_bins']
bins = np.arange(0, 2.5, 0.25)
num_bins = np.size(bins) - 1
bin_centers = (bins[:-1] + bins[1:]) / 2  # For plotting on x-axis

 num_bins = np.size(bins)-1
 peakt_by_tau_avg  = np.zeros(num_bins)
 peakt_by_tau_sem  = np.zeros(num_bins)
 peakt_by_tau_expt = [None]*num_bins
 peakm_by_tau_avg  = np.zeros(num_bins)
 peakm_by_tau_sem  = np.zeros(num_bins)
 peakm_by_tau_expt = [None]*num_bins
 
 for iBin in range(num_bins):
     # idx     = np.logical_and(is_good_tau==1,np.logical_and(tau>bins[iBin], tau<=bins[iBin+1]))
     # idx     = np.logical_and(is_good_tau==1,np.logical_and(is_stimd==0,np.logical_and(is_sig==1,idx)))
     
     
     idx     = np.logical_and(tau>bins[iBin], tau<=bins[iBin+1])
     # idx     = np.logical_and(is_good_tau==1,idx)
     # idx     = np.logical_and(is_good_SNR==True,idx)

     idx     = np.logical_and(is_sig==1,idx)
     idx     = np.logical_and(is_stimd==0,idx)
     
     
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
 
plt.figure()
plt.plot(peakt_by_tau_avg)
# %%
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
    peak_idx    = np.logical_and(peak_ts>tau_t_bins[iBin], peak_ts<=tau_t_bins[iBin+1])  # using wrong bins
    
    short_tau_t = np.logical_and(peak_idx,short_tau_nonstimd)
    long_tau_t  = np.logical_and(peak_idx,long_tau_nonstimd)
    
    t_counts[iBin][0]           = np.sum(np.logical_and(peak_idx==1,short_tau_stimd==1))
    t_counts[iBin][1]           = np.sum(np.logical_and(peak_idx==1,long_tau_stimd==1))
    
    tau_mat_t_counts[iBin][0,0] = np.sum(np.logical_and(short_tau_stimd,short_tau_t))
    tau_mat_t_counts[iBin][0,1] = np.sum(np.logical_and(short_tau_stimd,long_tau_t))
    tau_mat_t_counts[iBin][1,0] = np.sum(np.logical_and(long_tau_stimd,short_tau_t))
    tau_mat_t_counts[iBin][1,1] = np.sum(np.logical_and(long_tau_stimd,long_tau_t))
    
# count responding cells for normalization
ct_long  = np.sum(long_tau_nonstimd_denominator==1)
ct_short = np.sum(short_tau_nonstimd_denominator==1)
    
# divide by counts to get average
tau_mat[0,:] = tau_mat_counts[0,:] / ct_short
tau_mat[1,:] = tau_mat_counts[1,:] / ct_long

for iBin in range(len(tau_mat_t)):
    tau_mat_t[iBin][0,:] = tau_mat_t_counts[iBin][0,:] / t_counts[iBin][0]
    tau_mat_t[iBin][1,:] = tau_mat_t_counts[iBin][1,:] / t_counts[iBin][1]
    tau_mat_t_by_overall[iBin][0,:] = tau_mat_t_counts[iBin][0,:] / ct_short
    tau_mat_t_by_overall[iBin][1,:] = tau_mat_t_counts[iBin][1,:] / ct_long


# %%
unique_sess = list(np.unique(sess_ids))
ct_short    = 0
ct_long     = 0
tau_mat_counts       = np.zeros((2,2))
tau_mat              = np.zeros((2,2))
tau_t_bins           = opto_params['tau_by_time_bins']


### NO THESE ARE ALL INITIALIZED AS shared References!! Avoid in future will lead to each array being exactly the same even after setting
#new values

# tau_mat_t_counts     = [np.zeros((2,2))]*(len(tau_t_bins)-1)
# tau_mat_t            = [np.zeros((2,2))]*(len(tau_t_bins)-1) # normed by cells in each bin
# tau_mat_t_by_overall = [np.zeros((2,2))]*(len(tau_t_bins)-1) # normed by overall responding cells
# t_counts             = [np.zeros(2)]*(len(tau_t_bins)-1)   


tau_mat_t_counts     = [np.zeros((2,2)) for _ in range(len(tau_t_bins)-1)]
tau_mat_t            = [np.zeros((2,2)) for _ in range(len(tau_t_bins)-1)] # normed by cells in each bin
tau_mat_t_by_overall = [np.zeros((2,2)) for _ in range(len(tau_t_bins)-1)] # normed by overall responding cells

t_counts = [np.zeros(2) for _ in range(len(tau_t_bins)-1)]

# %%

tau_mat_counts_all     = [np.zeros((2,2)) for _ in range(len(unique_sess))]

for sess in unique_sess:
    
    unique_stim = list(np.unique(stim_ids[sess_ids==sess]))
    
    tau_mat              = np.zeros((2,2))
    
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
    
    
    tau_mat_counts_all[sess] = tau_mat_counts.copy()
    
    
    # # now do it by time by restricting by peak time
    # for iBin in range(len(tau_mat_t_counts)):
    #     peak_idx    = np.logical_and(peak_ts>tau_t_bins[iBin], peak_ts<=tau_t_bins[iBin+1])
    #     short_idx_t = np.logical_and(peak_idx,short_idx)
    #     long_idx_t  = np.logical_and(peak_idx,long_idx)
    #     total_t     = np.sum(np.logical_and(peak_idx,these_cells))-np.sum(stimd_idx)
    #     tau_mat_t[iBin][mat_row,0] += np.sum(short_idx_t==1) / total_t
    #     tau_mat_t[iBin][mat_row,1] += np.sum(long_idx_t==1) / total_t
    #     tau_mat_t_by_overall[iBin][mat_row,0] += np.sum(short_idx_t==1) / total_good_cells
    #     tau_mat_t_by_overall[iBin][mat_row,1] += np.sum(long_idx_t==1) / total_good_cells
    #     tau_mat_t_counts[iBin][mat_row,0] += np.sum(short_idx_t==1) 
    #     tau_mat_t_counts[iBin][mat_row,1] += np.sum(long_idx_t==1)
        
# multiply by counts to get average numbers (from average probs)
# tau_mat[0,:] = tau_mat[0,:] / ct_short
# tau_mat[1,:] = tau_mat[1,:] / ct_long
# tau_mat_counts[0,:] = tau_mat_counts[0,:] / ct_short
# tau_mat_counts[1,:] = tau_mat_counts[1,:] / ct_long


# for iBin in range(len(tau_mat_t)):
#     tau_mat_t[iBin][0,:] = tau_mat_t[iBin][0,:] / ct_short
#     tau_mat_t[iBin][1,:] = tau_mat_t[iBin][1,:] / ct_long
#     tau_mat_t_counts[iBin][0,:] = tau_mat_t_counts[iBin][0,:] / ct_short
#     tau_mat_t_counts[iBin][1,:] = tau_mat_t_counts[iBin][1,:] / ct_long
#     tau_mat_t_by_overall[iBin][0,:] = tau_mat_t_by_overall[iBin][0,:] / ct_short
#     tau_mat_t_by_overall[iBin][1,:] = tau_mat_t_by_overall[iBin][1,:] / ct_long