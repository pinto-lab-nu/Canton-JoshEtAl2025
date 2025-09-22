# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:49:37 2025

@author: jec822
"""

# %% ===========================================
# ===== Fig 4: evoked timecourse comparison ====
# ==============================================

fig4_params = {'sess_eg_v1' : 10,
               'stim_eg_v1' : 6,
               # 'sess_eg_v1' : 8,
               # 'stim_eg_v1' : 3,
               # 'sess_eg_m2' : 0,
               # 'stim_eg_m2' : 2}
                'sess_eg_m2' : 2,
                'stim_eg_m2' : 4}

fig = plt.figure(figsize=(8, 10))



# %%
# fig 4d roi-wise dff timecourse 

_, _,a= analyzeEvoked2P.plot_resp_fov('V1', 
                                      which_sess=fig4_params['sess_eg_v1'], 
                                      which_stim=fig4_params['stim_eg_v1'], 
                                      expt_type='standard', 
                                      resp_type='dff', 
                                      plot_what='peak_time', 
                                      prctile_cap=[0,98], 
                                      signif_only=False, 
                                      highlight_signif=True, 
                                      axis_handle=None,
                                      max_min=(0,8))

_, _,a= analyzeEvoked2P.plot_resp_fov('M2', 
                                      which_sess=fig4_params['sess_eg_m2'], 
                                      which_stim=fig4_params['stim_eg_m2'], 
                                      expt_type='standard', 
                                      resp_type='dff', 
                                      plot_what='peak_time', 
                                      prctile_cap=[0,98], 
                                      signif_only=False, 
                                      highlight_signif=True, 
                                      axis_handle=None,
                                      max_min=(0,8))