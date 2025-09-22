# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:48:39 2025

@author: jec822
"""



# %% ===========================================
# ===== Fig 3E_F
# ==============================================

fig3_params = {'sess_eg_v1' : 10,
               'stim_eg_v1' : 6,
               # 'sess_eg_v1' : 8,
               # 'stim_eg_v1' : 3,
               # 'sess_eg_m2' : 0,
               # 'stim_eg_m2' : 2}
                'sess_eg_m2' : 2,
                'stim_eg_m2' : 4}

fig = plt.figure(figsize=(8, 10))



_,_,a= analyzeEvoked2P.plot_resp_fov('V1', 
                                     which_sess=fig3_params['sess_eg_v1'], 
                                     which_stim=fig3_params['stim_eg_v1'], 
                                     expt_type='standard', 
                                     resp_type='dff', 
                                     plot_what='peak_mag', 
                                     prctile_cap=[0,98], 
                                     signif_only=False, 
                                     highlight_signif=True, 
                                     axis_handle=None,
                                     max_min=[-4,4])

# %%
_, _,a = analyzeEvoked2P.plot_resp_fov('M2', 
                                     which_sess=fig3_params['sess_eg_m2'], 
                                     which_stim=fig3_params['stim_eg_m2'], 
                                     expt_type='standard', 
                                     resp_type='dff', 
                                     plot_what='peak_mag', 
                                     prctile_cap=[0,98], 
                                     signif_only=False, 
                                     highlight_signif=True, 
                                     axis_handle=None,
                                     max_min=[-4,4])




