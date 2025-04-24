# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:00:00 2025

@author: jec822
"""

# def get_all_tau(area, params=params, dff_type='residuals_dff'):

   area="M2"
   params=opto_params
   dff_type='residuals_dff'
  
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
       # keys.append((spont_timescales.TwopTauInclusion & primary_key & 'is_good_tau_roi=1').fetch('KEY'))
       # keys.append((spont_timescales.TwopTauInclusion & primary_key & 'is_good_tau_roi=0').fetch('KEY'))
       keys.append((spont_timescales.TwopTauInclusion & primary_key).fetch('KEY'))

   keys = [entry for sublist in keys for entry in sublist]  # flatten

   # fetch taus and associated keys, then filter for tau > 0.1
   tau_entries = (spont_timescales.TwopTau & keys).fetch('tau', 'KEY')
   
   aicc_double = (spont_timescales.TwopTau & keys).fetch('aicc_double', 'KEY')
   r2_fit_double = (spont_timescales.TwopTau & keys).fetch('r2_fit_double', 'KEY')
   acorr,lags  = (spont_timescales.TwopAutocorr & keys).fetch('autocorr_vals','lags_sec')
   
   r_fit_single = (spont_timescales.TwopTau & keys).fetch('r2_fit_single', 'KEY')
   
   is_good = np.array((spont_timescales.TwopTauInclusion & keys).fetch('is_good_tau_roi'))
   
   bic_single = (spont_timescales.TwopTau & keys).fetch('bic_single', 'KEY')
   bic_double = (spont_timescales.TwopTau & keys).fetch('bic_double', 'KEY')
   
   tau_single=(spont_timescales.TwopTau & keys).fetch('single_fit_params', 'KEY')


   tau_double=(spont_timescales.TwopTau & keys).fetch('double_fit_params', 'KEY')
   
   
   taus_all, keys_all = tau_entries
   mask = taus_all > 0.0
   taus = taus_all[mask]
   keys = [keys_all[i] for i in np.where(mask)[0]]

   # figure out how many of those are somas
   seg_keys = VM['twophoton'].Segmentation2P & keys
   is_soma = np.array((VM['twophoton'].Roi2P & seg_keys).fetch('is_soma'))
   total_soma = np.sum(is_soma)

   end_time = time.time()
   print("     done after {: 1.1f} min".format((end_time - start_time) / 60))

   return taus, keys, total_soma
# %%

bic_d =bic_double[0]
bic_s=bic_single[0]

a=bic_d<bic_s
# double=


mask = r2_fit_double[0] > 0.9
double= r2_fit_double[0][mask]


mask = r_fit_single[0] > 0.9
single= r_fit_single[0][mask]



mask = (r2_fit_double[0] > 0.9) & (r_fit_single[0]<r2_fit_double[0])

mask2 = (r_fit_single[0] > 0.9) & (r_fit_single[0]>r2_fit_double[0])


# %%
r2_s=r_fit_single[0]
r2_d=r2_fit_double[0]
# r2_s, r2_d = (TwopTau & key).fetch1('r2_fit_single','r2_fit_double')


r2_s = r_fit_single[0]
r2_d = r2_fit_double[0]
r2 = r2_s if r2_s > r2_d else r2_d


if r2_s > r2_d:
    r2 = r2_s
else:
    r2 = r2_d
    # %%
    
r2_s = r_fit_single[0]
r2_d = r2_fit_double[0]
# r2 = max(r2_s, r2_d)


b=(r2_s < r2_d) & (r2_d> 0.9)

a=(r2_s > r2_d) & (r2_s> 0.9)

c= (r2_s> 0.9)
# %%

import numpy as np
import matplotlib.pyplot as plt


cell=7
# Time axis
t = np.linspace(0, 29, 900)

fit_params=tau_double[0][cell]

# Extract parameters
A0 = fit_params['A_fit_0']
tau0 = fit_params['tau_fit_0']
A1 = fit_params['A_fit_1']
tau1 = fit_params['tau_fit_1']
offset = fit_params['offset_fit_dual']

# Compute the fitted curve
fit_curve = A0 * np.exp(-t / tau0) + A1 * np.exp(-t / tau1) + offset

# Plot
plt.plot(t, fit_curve, label='Dual exponential fit')
plt.xlabel('Time')
plt.ylabel('Fitted Value')
plt.legend()
plt.show()


# Time axis
# t = np.linspace(0, 10, 1000)
t = np.linspace(0, 29, 900)

fit_params_mono=tau_single[0][cell]

# Extract parameters
A = fit_params_mono['A_fit_mono']
tau = fit_params_mono['tau_fit_mono']
offset = fit_params_mono['offset_fit_mono']

# Compute mono-exponential fit
fit_curve = A * np.exp(-t / tau) + offset

# Plot
plt.plot(t, fit_curve, label='Mono-exponential fit', color='orange')
plt.xlabel('Time')
plt.ylabel('Fitted Value')
plt.legend()
plt.show()

plt.plot(t,acorr[cell][0:len(t)])
