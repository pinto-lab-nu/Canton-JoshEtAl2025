# -*- coding: utf-8 -*-
"""
Created on Tue May  6 17:03:26 2025

@author: jec822
"""

# %%

v1_all_distance_matrices = PCA_functions.compute_3d_transition_distances(v1_pca_results,num_pcs=3)
m2_all_distance_matrices = PCA_functions.compute_3d_transition_distances(m2_pca_results,num_pcs=3)


v1_distances_flat = PCA_functions.plot_distance_histogram(v1_all_distance_matrices)
m2_distances_flat = PCA_functions.plot_distance_histogram(m2_all_distance_matrices)

analysis_plotting_functions.plot_ecdf_comparison(v1_distances_flat, m2_distances_flat,
                     label1='V1', label2='M2',title='ECDF norm stimulation deviation first time point')


# %%

max_indices, response_sems = PCA_functions.compute_index_of_max_poststim_distance_from_baseline(m2_pca_results, num_pcs=10)

sem_matrices = PCA_functions.compute_sem_from_max_index_matrices(max_indices)

m2_flat_sem = np.array([sem for exp in sem_matrices for sem in exp])

max_indices, response_sems = PCA_functions.compute_index_of_max_poststim_distance_from_baseline(v1_pca_results, num_pcs=10)

sem_matrices = PCA_functions.compute_sem_from_max_index_matrices(max_indices)

v1_flat_sem = np.array([sem for exp in sem_matrices for sem in exp])

analysis_plotting_functions.plot_ecdf_comparison(v1_flat_sem, m2_flat_sem,
                     label1='V1', label2='M2',title='ECDF SEM of indices of max deviation by expt')

# %%
# v1_all_max_distances = PCA_functions.compute_normalized_max_deviation(m2_pca_results_scramble)

v1_all_max_distances = PCA_functions.compute_normalized_max_deviation(v1_pca_results)
m2_all_max_distances = PCA_functions.compute_normalized_max_deviation(m2_pca_results)

sem_matrices = PCA_functions.compute_sem_from_max_index_matrices(v1_all_max_distances)

m2_flat_sem = PCA_functions.np.array([sem for exp in sem_matrices for sem in exp])

sem_matrices = PCA_functions.compute_sem_from_max_index_matrices(m2_all_max_distances)

v1_flat_sem = np.array([sem for exp in sem_matrices for sem in exp])


analysis_plotting_functions.plot_ecdf_comparison(v1_flat_sem, m2_flat_sem,
                     label1='V1', label2='M2', title='ECDF SEM of max deviation by expt')


# %% Max distance from baseline period

v1_all_max_distances = PCA_functions.compute_max_poststim_distance_from_baseline_3d(v1_pca_results)
# v1_all_max_distances = PCA_functions.compute_max_poststim_distance_from_baseline_3d(m2_pca_results_scramble)
m2_all_max_distances = PCA_functions.compute_max_poststim_distance_from_baseline_3d(m2_pca_results)

# v1_all_max_distances = PCA_functions.compute_normalized_max_deviation(v1_pca_results)
# m2_all_max_distances = PCA_functions.compute_normalized_max_deviation(m2_pca_results)


v1_distances_flat = PCA_functions.plot_distance_histogram(v1_all_max_distances)
m2_distances_flat = PCA_functions.plot_distance_histogram(m2_all_max_distances)

analysis_plotting_functions.plot_ecdf_comparison(v1_distances_flat, m2_distances_flat,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='ECDF max deviation post-stim in 3D PC space by expt',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Max distance from baseline",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=8, xticks_step=2,
                     yticks_start=0, yticks_end=1, yticks_step=0.2,
                     xlim=[0,50],
                     stat_test='auto',
                     figsize=[4,5],
                     show_normality_pvals=True)


# %%

v1_all_max_distances = PCA_functions.compute_index_of_max_poststim_distance_from_baseline_3d(v1_pca_results)
m2_all_max_distances = PCA_functions.compute_index_of_max_poststim_distance_from_baseline_3d(m2_pca_results)

v1_distances_flat = PCA_functions.plot_distance_histogram(v1_all_max_distances)
m2_distances_flat = PCA_functions.plot_distance_histogram(m2_all_max_distances)

analysis_plotting_functions.plot_ecdf_comparison(v1_distances_flat, m2_distances_flat,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='ECDF  index of max deviation post-stim in 3D PC space by expt',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Time of max distance from baseline")

# %% Path curvature analysis

v1_path_length_and_curvature = PCA_functions.compute_path_length_and_curvature(v1_pca_results)

# v1_path_length_and_curvature = PCA_functions.compute_path_length_and_curvature(m2_pca_results_scramble)

m2_path_length_and_curvature = PCA_functions.compute_path_length_and_curvature(m2_pca_results)

v1_flat_response_path_lengths = PCA_functions.extract_metric_from_metrics_all(v1_path_length_and_curvature, 'response_path_length')
m2_flat_response_path_lengths = PCA_functions.extract_metric_from_metrics_all(m2_path_length_and_curvature, 'response_path_length')


v1_flat_baseline_path_lengths = PCA_functions.extract_metric_from_metrics_all(v1_path_length_and_curvature, 'baseline_path_length')
m2_flat_baseline_path_lengths = PCA_functions.extract_metric_from_metrics_all(m2_path_length_and_curvature, 'baseline_path_length')


v1_flat_response_curvature = PCA_functions.extract_metric_from_metrics_all(v1_path_length_and_curvature, 'response_curvature')
m2_flat_response_curvature = PCA_functions.extract_metric_from_metrics_all(m2_path_length_and_curvature, 'response_curvature')

v1_flat_baseline_curvature = PCA_functions.extract_metric_from_metrics_all(v1_path_length_and_curvature, 'baseline_curvature')
m2_flat_baseline_curvature = PCA_functions.extract_metric_from_metrics_all(m2_path_length_and_curvature, 'baseline_curvature')

# %% 

analysis_plotting_functions.plot_ecdf_comparison(v1_flat_response_path_lengths, m2_flat_response_path_lengths,
                     label1='VIsp', label2='MOs')

analysis_plotting_functions.plot_ecdf_comparison(v1_flat_baseline_path_lengths, m2_flat_baseline_path_lengths,
                      label1='VIsp', label2='MOs')


analysis_plotting_functions.plot_ecdf_comparison(v1_flat_response_curvature, m2_flat_response_curvature,
                      label1='VIsp', label2='MOs')

analysis_plotting_functions.plot_ecdf_comparison(v1_flat_baseline_curvature, m2_flat_baseline_curvature,
                      label1='VIsp', label2='MOs')

# %% Centroid distance comparison

# From previous function:
baseline_cents, poststim_cents, centroid_dists = PCA_functions.compute_centroid_distances(v1_pca_results, num_pcs=10)

# Compute pairwise distances:
baseline_pairwise_dists = PCA_functions.compute_pairwise_centroid_distances(baseline_cents)
poststim_pairwise_dists = PCA_functions.compute_pairwise_centroid_distances(poststim_cents)

# baseline_pairwise_dists = PCA_functions.compute_flat_upper_centroid_distances(baseline_cents)
# poststim_pairwise_dists = PCA_functions.compute_flat_upper_centroid_distances(poststim_cents)

baseline_pairwise_summary = PCA_functions.summarize_pairwise_distances(baseline_pairwise_dists)
poststim_pairwise_summary = PCA_functions.summarize_pairwise_distances(poststim_pairwise_dists)

# To flatten SEMs across all experiments/indices for example:
v1_flat_baseline_sems = [item for sublist in baseline_pairwise_summary['mean'] for item in sublist]


v1_baseline_cents_flat = PCA_functions.plot_distance_histogram(baseline_pairwise_dists)
v1_poststim_cents_flat = PCA_functions.plot_distance_histogram(poststim_pairwise_dists)

# plt.scatter(v1_baseline_cents_flat,v1_poststim_cents_flat)

a2=PCA_functions.plot_baseline_poststim_with_significance(
    baseline_pairwise_dists,
    poststim_pairwise_dists,
    xlim=[0,25],
    ylim=[0,100]
    )

# %% Centroid distance comparison M2

baseline_cents, poststim_cents, centroid_dists = PCA_functions.compute_centroid_distances(m2_pca_results, num_pcs=10,distance_type='euclidean')

# Compute pairwise distances:
baseline_pairwise_centroid_dists = PCA_functions.compute_pairwise_centroid_distances(baseline_cents)
poststim_pairwise_centroid_dists = PCA_functions.compute_pairwise_centroid_distances(poststim_cents)

# baseline_pairwise_centroid_dists = PCA_functions.compute_flat_upper_centroid_distances(baseline_cents)
# poststim_pairwise_centroid_dists = PCA_functions.compute_flat_upper_centroid_distances(poststim_cents)

baseline_pairwise_summary = PCA_functions.summarize_pairwise_distances(baseline_pairwise_centroid_dists)
poststim_pairwise_summary = PCA_functions.summarize_pairwise_distances(poststim_pairwise_centroid_dists)

# To flatten SEMs across all experiments/indices for example:
m2_flat_baseline_sems = [item for sublist in baseline_pairwise_summary['mean'] for item in sublist]

a=PCA_functions.plot_baseline_poststim_with_significance(
    baseline_pairwise_centroid_dists,
    poststim_pairwise_centroid_dists,
    xlim=[0,25],
    ylim=[0,100]
    )
    

# %%

analysis_plotting_functions.plot_ecdf_comparison(a['r'],a2['r'],
                     label1='M2', label2='V1')


analysis_plotting_functions.plot_ecdf_comparison(np.array(v1_flat_baseline_sems), np.array(m2_flat_baseline_sems),
                     label1='V1', label2='M2')

# %%

v1_d = PCA_functions.plot_distance_histogram(v1_all_max_distances)
m2_d = PCA_functions.plot_distance_histogram(m2_all_max_distances)


# %% Using point by point distance

all_baseline_dists,all_poststim_dists = compute_pairwise_pointwise_distances(m2_pca_results,num_pcs=10,distance_type='manhattan',reduction='mean')
# all_baseline_dists,all_poststim_dists = PCA_functions.compute_pairwise_trial_distances_static(m2_pca_results,num_pcs=10,mode='static')

# dists = PCA_functions.compute_pointwise_distances(m2_pca_results)

all_baseline_dists_upper = PCA_functions.extract_upper_triangle_nested(all_baseline_dists)
all_poststim_dists_upper = PCA_functions.extract_upper_triangle_nested(all_poststim_dists)

v1_d = PCA_functions.plot_distance_histogram(all_baseline_dists_upper)
v1_d_post = PCA_functions.plot_distance_histogram(all_poststim_dists_upper)

plt.figure()
plt.scatter(v1_d,v1_d_post)
# plt.ylim((0,12000))

analysis_plotting_functions.plot_ecdf_comparison(v1_d, v1_d_post,
                     label1='V1_pre', label2='V1_post')

a=PCA_functions.plot_baseline_poststim_with_significance(
    all_baseline_dists_upper,
    all_poststim_dists_upper,
    # xlim=[0,250],
    # ylim=[0,1000]
    )

# %% Using point by point distance

all_baseline_dists,all_poststim_dists = PCA_functions.compute_pairwise_pointwise_distances(v1_pca_results,num_pcs=10,distance_type='manhattan',reduction='mean')
# all_baseline_dists,all_poststim_dists = PCA_functions.compute_pairwise_trial_distances_static(v1_pca_results,num_pcs=10,mode='static')

# dists = PCA_functions.compute_pointwise_distances(m2_pca_results)

all_baseline_dists_upper = PCA_functions.extract_upper_triangle_nested(all_baseline_dists)
all_poststim_dists_upper = PCA_functions.extract_upper_triangle_nested(all_poststim_dists)


m2_d = PCA_functions.plot_distance_histogram(all_baseline_dists_upper)
m2_d_post = PCA_functions.plot_distance_histogram(all_poststim_dists_upper)

plt.figure()
plt.scatter(v1_d,v1_d_post)
# plt.ylim((0,12000))

analysis_plotting_functions.plot_ecdf_comparison(m2_d, m2_d_post,
                     label1='M2_pre', label2='M2_post')

a2=PCA_functions.plot_baseline_poststim_with_significance(
    all_baseline_dists_upper,
    all_poststim_dists_upper,
    # xlim=[0,250],
    # ylim=[0,1000]
    )

# %%

analysis_plotting_functions.plot_ecdf_comparison(v1_d, m2_d,
                     label1='V1_pre', label2='M2_pre',
                     stat_test='auto')

analysis_plotting_functions.plot_ecdf_comparison(v1_d_post, m2_d_post,
                     label1='V1_post', label2='M2_post',
                     stat_test='auto')

# %%

all_correlations = PCA_functions.correlate_baseline_poststim_dists(
    all_baseline_dists_upper,
    all_poststim_dists_upper
)

# Flatten to a 2-column array: [r, p]
flat_correlation_matrix = np.array([
    [r, p]
    for expt in all_correlations
    for (r, p) in expt
    if not (np.isnan(r) or np.isnan(p))
])

all_correlations_by_expt,points  = PCA_functions.correlate_per_experiment_flat(
    all_baseline_dists_upper,
    all_poststim_dists_upper
)

all_correlations_by_expt = np.array(correlate_per_experiment_flat(all_baseline_dists_upper, all_poststim_dists_upper)[0])
all_correlations_by_expt = np.array([list(t) for t in all_correlations_by_expt])

# %%

# all_baseline_dists_upper = extract_upper_triangle_nested(baseline_pairwise_centroid_dists)
# all_poststim_dists_upper = extract_upper_triangle_nested(poststim_pairwise_centroid_dists)


# %% This compares distances between trials , mainly a tool to help me find examples to plot for 5A

trial_pairs_different = find_trial_pairs_by_threshold(
    all_baseline_dists,
    all_poststim_dists,
    baseline_thresh=0.8,
    poststim_thresh=0.6,
    baseline_mode='above',
    poststim_mode='above',
    all_pointwise_dists=dists,
    pointwise_thresh=0.6,
    pointwise_mode='above'
)

trial_pairs_similar = find_trial_pairs_by_threshold(
    all_baseline_dists,
    all_poststim_dists,
    baseline_thresh=0.1,
    poststim_thresh=0.1,
    baseline_mode='below',
    poststim_mode='below',
    all_pointwise_dists=dists,
    pointwise_thresh=0.6,
    pointwise_mode='above'
)


# trial_pairs_different = find_trial_pairs_by_absolute_threshold(
#     all_baseline_dists,
#     all_poststim_dists,
#     baseline_thresh=1000,
#     poststim_thresh=2000,
#     baseline_mode='above',
#     poststim_mode='above',
#     all_pointwise_dists=dists,
#     pointwise_thresh=0.9,
#     pointwise_mode='above'
# )

trial_pairs_similar = find_trial_pairs_by_absolute_threshold(
    all_baseline_dists,
    all_poststim_dists,
    baseline_thresh=1000,
    poststim_thresh=4000,
    baseline_mode='below',
    poststim_mode='below',
    all_pointwise_dists=dists,
    pointwise_thresh=0.9,
    pointwise_mode='above'
)

triplets = find_similar_different_triplets(
    trial_pairs_similar=trial_pairs_similar,
    trial_pairs_different=trial_pairs_different
)

# %% centroid dists

trial_pairs_different = find_trial_pairs_by_threshold(
    baseline_pairwise_centroid_dists,
    poststim_pairwise_centroid_dists,
    baseline_thresh=0.7,
    poststim_thresh=0.7,
    baseline_mode='above',
    poststim_mode='above',
    # all_pointwise_dists=dists,
    pointwise_thresh=0.9,
    pointwise_mode='above'
)

trial_pairs_similar = find_trial_pairs_by_threshold(
    baseline_pairwise_centroid_dists,
    poststim_pairwise_centroid_dists,
    baseline_thresh=0.2,
    poststim_thresh=0.2,
    baseline_mode='below',
    poststim_mode='below',
    # all_pointwise_dists=dists,
    pointwise_thresh=0.9,
    pointwise_mode='above'
)

# trial_pairs_different = find_trial_pairs_by_absolute_threshold(
#     baseline_pairwise_centroid_dists,
#     poststim_pairwise_centroid_dists,
#     baseline_thresh=2,
#     poststim_thresh=8,
#     baseline_mode='above',
#     poststim_mode='above',
#     all_pointwise_dists=dists,
#     pointwise_thresh=0.9,
#     pointwise_mode='above'
# )

# trial_pairs_similar = find_trial_pairs_by_absolute_threshold(
#     baseline_pairwise_centroid_dists,
#     poststim_pairwise_centroid_dists,
#     baseline_thresh=1,
#     poststim_thresh=2,
#     baseline_mode='below',
#     poststim_mode='below',
#     all_pointwise_dists=dists,
#     pointwise_thresh=0.9,
#     pointwise_mode='above'
# )

triplets = find_similar_different_triplets(
    trial_pairs_similar=trial_pairs_similar,
    trial_pairs_different=trial_pairs_different
)

# %%
baseline_vectors, poststim_vectors = compute_pairwise_pointwise_vectors(v1_pca_results, num_pcs=3)


# %%

n_pcs = 10  # Number of principal components you want to extract
m2_cum_var_matrix = extract_cum_var_explained_matrix(m2_pca_results, n_pcs)
v1_cum_var_matrix = extract_cum_var_explained_matrix(v1_pca_results, n_pcs)


# cum_var_matrix shape: (num_experiments, n_pcs)
plot_two_cum_var_explained_with_sem(v1_cum_var_matrix, m2_cum_var_matrix, label1='V1', label2='M2',title='ECDF cumulative variance explained by PCs')



# %% Looking at Covariance

analysis_plotting_functions.plot_ecdf_comparison(np.array(v1_pca_results['corr_by_expt']), np.array(m2_pca_results['corr_by_expt']),
                     label1='Condition A', label2='Condition B',title='ECDF cumulative covariance by expt')

plt.scatter(np.array(m2_pca_results['corr_by_expt']),np.array(m2_pca_results['p_by_expt']))
# %%

PCA_functions.histogram_colored_by_pvalue(
    np.array(m2_pca_results['corr_by_expt']),
    np.array(m2_pca_results['p_by_expt']),
    bins=10,
    cmap='Blues',
    center_bins_around_zero=True
)

# %%
PCA_functions.histogram_color_by_significant_p(
    np.array(m2_pca_results['corr_by_expt']),
    np.array(m2_pca_results['p_by_expt']),
    bins=10,
    cmap='Blues',
    center_bins_around_zero=True
)


# %%
PCA_functions.histogram_color_by_significant_proportion(
    np.array(v1_pca_results['corr_by_expt']),
    np.array(v1_pca_results['p_by_expt']),
    bins=15,
    cmap='Blues',
    center_bins_around_zero=True
)
# %%
PCA_functions.overlay_significant_histogram(
    np.array(m2_pca_results['corr_by_expt']),
    np.array(m2_pca_results['p_by_expt']),
    bins=10,
    center_bins_around_zero=True,
    bar_gap=0.1,
    median_label="median",
    ylim=(0,8), 
    xlim=(-0.4,0.4),
    bar_width=.05
)

# %%

PCA_functions.overlay_significant_histogram(
    np.array(v1_pca_results['corr_by_expt']),
    np.array(v1_pca_results['p_by_expt']),
    bins=5,
    center_bins_around_zero=True,
    bar_gap=10,
    median_label="median",
    ylim=(0,8),
    xlim=(-0.4,0.4),
    bar_width=.05

)
# %%

PCA_functions.overlay_significant_histogram(
    np.array(flat_correlation_matrix[:,0]),
    np.array(flat_correlation_matrix[:,1]),
    bins=10,
    center_bins_around_zero=True,
    bar_gap=0.1,
    median_label="median"
)
# %%
PCA_functions.overlay_significant_histogram(
    np.array(all_correlations_by_expt[:,0]),
    np.array(all_correlations_by_expt[:,1]),
    bins=10,
    center_bins_around_zero=True,
    bar_gap=0.1,
    median_label="median"
)



  
# %% PLots for r values by stim and expt type

# This is looking across stimulation, comparing stim 1 to stim 2 etc.
analysis_plotting_functions.plot_ecdf_comparison(v1_pca_results['stim_pair_r'], m2_pca_results['stim_pair_r'],
                     label1='V1', label2='M2',title='',
                     xlabel='r values')

analysis_plotting_functions.plot_ecdf_comparison(v1_pca_results['stim_pair_p'], m2_pca_results['stim_pair_p'],
                     label1='V1', label2='M2',title='ECDF norm stimulation deviation first time point')

# This is looking within stims

analysis_plotting_functions.plot_ecdf_comparison(v1_pca_results['stim_trial_r'], m2_pca_results['stim_trial_r'],
                     label1='V1', label2='M2',title='',
                     xlabel='r values')


analysis_plotting_functions.plot_ecdf_comparison(v1_pca_results['stim_trial_p'], m2_pca_results['stim_trial_p'],
                     label1='V1', label2='M2',title='ECDF norm stimulation deviation first time point')



# %% This is looking across stimulation, comparing stim 1 to stim 2 etc.

PCA_functions.overlay_significant_histogram(
    np.array(v1_pca_results['stim_pair_r']),
    np.array(v1_pca_results['stim_pair_p']),
    bins=15,
    center_bins_around_zero=True,
    bar_gap=.1,
    median_label="median",
    ylim=(0,250),
    xlim=(-1,1),
    # bar_width=.05
)

PCA_functions.overlay_significant_histogram(
    np.array(m2_pca_results['stim_pair_r']),
    np.array(m2_pca_results['stim_pair_p']),
    bins=15,
    center_bins_around_zero=True,
    bar_gap=.1,
    median_label="median",
    ylim=(0,250),
    xlim=(-1,1),
    # bar_width=.05
)
# %%  Figure 5D
# This is looking within stims

PCA_functions.overlay_significant_histogram(
    np.array(v1_pca_results['stim_trial_r']),
    np.array(v1_pca_results['stim_trial_p']),
    bins=15,
    center_bins_around_zero=True,
    bar_gap=.1,
    median_label="median",
    ylim=(0,40),
    xlim=(-1,1),
    # bar_width=.05
)

PCA_functions.overlay_significant_histogram(
    np.array(m2_pca_results['stim_trial_r']),
    np.array(m2_pca_results['stim_trial_p']),
    bins=15,
    center_bins_around_zero=True,
    bar_gap=.1,
    median_label="median",
    ylim=(0,40),
    xlim=(-1,1),
    # bar_width=.05
)