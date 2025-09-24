# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:38:54 2025

@author: jec822
"""

# %%
# opsin expression
expression_stats, _ = analyzeEvoked2P.plot_opsin_expression_vs_response(params=opto_params, 
                                                                         expt_type='standard', 
                                                                         resp_type='dff',
                                                                         plot_what='stimd', 
                                                                         axis_handle=None)

# %% Figure S2 C D
_, _ = analyzeEvoked2P.plot_opsin_expression_vs_response(params=opto_params, 
                                                        expt_type='standard', 
                                                        resp_type='dff',
                                                        plot_what='stimd', 
                                                        v1_data=expression_stats['V1_summary'],
                                                        m2_data=expression_stats['M2_summary'],
                                                        axis_handle=None,
                                                        xlim=(-3, 3), ylim=(-1, 30),
                                                        figsize=(4,4)
                                                        )

_, _ = analyzeEvoked2P.plot_opsin_expression_vs_response(params=opto_params, 
                                                        expt_type='standard', 
                                                        resp_type='dff',
                                                        plot_what='non_stimd', 
                                                        v1_data=expression_stats['V1_summary'],
                                                        m2_data=expression_stats['M2_summary'],
                                                        axis_handle=None,
                                                        xlim=(-3, 3), ylim=(-1, 30),
                                                        figsize=(4,4)
