

# %% =========================================
# ===== Fig 2: spontaneous 2p timescales =====
# ============================================
fig = plt.figure(figsize=(8, 10))


# # Analyzing data
v1_taus, v1_keys, v1_total = analyzeSpont2P.get_all_tau('V1', params=tau_params, dff_type='residuals_dff')
m2_taus, m2_keys, m2_total = analyzeSpont2P.get_all_tau('M2', params=tau_params, dff_type='residuals_dff')




# Fig 2E: 2p tau summary
tau_stats, ax_tau = analyzeSpont2P.plot_area_tau_comp(v1_taus=v1_taus, m2_taus=m2_taus, axis_handle=None, params=tau_params,xlim=[0,10])
tau_stats['V1_total_num_cells'] = v1_total
tau_stats['M2_total_num_cells'] = m2_total
# ax.text(-0.1, 1.05, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')



# Fig 2F: Clustering FOV example
v1_centr, v1_rec_ids = analyzeSpont2P.get_centroids_by_rec(v1_keys)
m2_centr, m2_rec_ids = analyzeSpont2P.get_centroids_by_rec(m2_keys)

# ax = row2_axs[0]
fov_v1 = analyzeSpont2P.plot_tau_fov(v1_keys, v1_rec_ids, which_sess=9, do_zscore=False, prctile_cap=[0, 95], axis_handle=None, fig_handle=fig,max_min=[0.25,3],cmap='Blues')
# ax.text(-0.1, 1.05, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')


# ax = row2_axs[1]
# for i in range(15,30):
#     fov_m2 = analyzeSpont2P.plot_tau_fov(m2_keys, m2_rec_ids, which_sess=i, do_zscore=False, prctile_cap=[0, 95], axis_handle=None, fig_handle=fig)
fov_m2 = analyzeSpont2P.plot_tau_fov(m2_keys, m2_rec_ids, which_sess=19, do_zscore=False, prctile_cap=[0, 95], axis_handle=None, fig_handle=fig,max_min=[0.25,6],cmap='viridis')


# Fig 2G: Clustering
# ax = row3_axs[0]
clust_stats_v1, tau_diff_mat_v1 = analyzeSpont2P.clustering_by_tau(v1_taus, v1_centr, v1_rec_ids, params=tau_params)
clust_stats_m2, tau_diff_mat_m2 = analyzeSpont2P.clustering_by_tau(m2_taus, m2_centr, m2_rec_ids, params=tau_params)
cax = analyzeSpont2P.plot_clustering_comp(v1_clust=clust_stats_v1, m2_clust=clust_stats_m2, params=tau_params, axis_handle=None)
# ax.text(-0.1, 1.05, 'D', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')


