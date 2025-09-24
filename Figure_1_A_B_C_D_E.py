import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

repo_dir = r"\Users\lai7370\Projects\expressionTauRegressions"
os.chdir(repo_dir)

from expression_script import *


args_dict = script_args(from_command_line=False)
args_dict['gene_limit'] = 2                     # use -1 for all transcripts, this will result in large predictor matricies that require ~180GB of memory for full regressions
args_dict['regressions_to_start'] = [0, 1, 3]   # numeric codes descrbed below

# Regression type codes:
# 0 : transcript expression                     -> AP CCF & ML CCF
# 1 : transcript expression                     -> tau
# 2 : transcript expression                     -> tau residuals
# 3 : transcript expression & AP CCF & ML CCF   -> tau


###############################
### Now run the regressions ###
###############################
# Allen merfish data and voxel annotations get downloaded into /Data, needs ~50GB free disk space.
# The first time running may take a while, as it needs to download data for all transcripts into the /Data directory. 
# The amount of time it takes to load data into memory is dependent on the gene_limit parameter, the script leverages the
# allen API to only load needed transcripts on-the-fly for faster loading times. Regression outputs get saved to /Plotting
run_transcript_tau_GLMs(args_dict)

### Make figures ###
glm_plotting_output_path = os.path.join(repo_dir, 'Plotting', 'Cux2-Ai96', 'pooling0.1mm', 'GenePredictors', 'Merfish-Imputed')
meta_dict = pickle.load(open(os.path.join(glm_plotting_output_path, f'plotting_data.pickle'), 'rb'))
used_merfish_genes_longnames = pickle.load(open(os.path.join(repo_dir, 'Plotting', f'merfishImputed_usedLongGeneNames.pickle'), 'rb'))
fig_1_b_merfish_data = pickle.load(open(os.path.join(repo_dir, 'Plotting', f'fig_1_b_merfish_data.pickle'), 'rb'))
fig_1_b_tau_data = pickle.load(open(os.path.join(repo_dir, 'Plotting', f'fig_1_b_tau_data.pickle'), 'rb'))
fig_1_a_data_dict = pickle.load(open(os.path.join(repo_dir, 'Plotting', f'fig_1_a_data_dict.pickle'), 'rb'))

# Cross layers codes (comparing model performance for merfish data isolated to different layers):
# 0 : L2/3
# 1 : L4/5
# 2 : L6

# plot_gene_enrichment() and plot_regressions() return lists of figure handles,
# the number of which are determined by how many layers are resquested in the output
# (e.g. cross_layers = [0, 2] will return two figures, one for each layer)

### Fig 1 A, tau vs AP CCF & ML CCF ###
def plot_fig_1_a(fig_1_a_data_dict):

    fig_1_a_ml_ap_tau, ax = plt.subplots(1,2,figsize=(5,2))
    
    ml_vals = fig_1_a_data_dict['ml_vals']
    ap_vals = fig_1_a_data_dict['ap_vals']
    tau_vals = fig_1_a_data_dict['tau_vals']
    colors = fig_1_a_data_dict['colors']
    struct_list = fig_1_a_data_dict['struct_list']
    area_colors = fig_1_a_data_dict['area_colors']
    r_correlation_ml = fig_1_a_data_dict['r_correlation_ml']
    p_correlation_ml = fig_1_a_data_dict['p_correlation_ml']
    r_correlation_ap = fig_1_a_data_dict['r_correlation_ap']
    p_correlation_ap = fig_1_a_data_dict['p_correlation_ap']

    ax[0].scatter(ml_vals, tau_vals, color=colors, s=0.4)
    ax[0].set_title(f'r={round(r_correlation_ml,3)}, p={p_correlation_ml:.3e}')
    ax[1].scatter(ap_vals, tau_vals, color=colors, s=0.4)
    ax[1].set_title(f'r={round(r_correlation_ap,3)}, p={p_correlation_ap:.3e}')
    ax[0].set_xlabel('ML CCF (mm)'), ax[1].set_xlabel('AP CCF (mm)')
    ax[0].set_ylabel(f'$\\tau$ (s)')
   
    region_color_patch = [mpatches.Patch(color=area_colors[i], label=struct_list[i]) for i in range(len(struct_list))]
    ax[1].legend(handles=region_color_patch, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', borderaxespad=0.)
    plt.close()

    return fig_1_a_ml_ap_tau

fig_1_a_ml_ap_tau = plot_fig_1_a(fig_1_a_data_dict)

### Fig 1 B, MERFISH & tau in CCF space ###
def plot_fig_1_b(fig_1_b_merfish_data, fig_1_b_tau_data):

    mlCCF_per_cell_H2layerFiltered = fig_1_b_merfish_data['mlCCF_per_cell_H2layerFiltered']
    apCCF_per_cell_H2layerFiltered = fig_1_b_merfish_data['apCCF_per_cell_H2layerFiltered']
    ml_filter = np.where(mlCCF_per_cell_H2layerFiltered['10'][0].ravel() > 2)[0]
    
    fig_1_b_merfish, ax = plt.subplots(1, 1, figsize=(1.75,3))
    plt.scatter(mlCCF_per_cell_H2layerFiltered['10'][0][ml_filter][::10,:], apCCF_per_cell_H2layerFiltered['10'][0][ml_filter][::10,:], color='black', s=0.1)
    plt.xticks(np.arange(2, 7, 1))
    plt.axis('equal')
    plt.close()

    fig_1_b_tau, ax = plt.subplots(1, 1, figsize=(2.5,3))
    cmap = plt.get_cmap('cool')
    sc = ax.scatter(fig_1_b_tau_data[0][:,0], fig_1_b_tau_data[0][:,1], c=fig_1_b_tau_data[0][:,2], s=2, cmap=cmap)
    ax.set_xlabel(r'L$\leftrightarrow$M (mm)'), ax.set_ylabel(r'P$\leftrightarrow$A (mm)'), ax.axis('equal')
    ax.set_xticks(np.arange(2, 6, 1))
    cbar = fig_1_b_tau.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label(f'$\\tau$ (s)')
    plt.close()
    
    return fig_1_b_merfish, fig_1_b_tau

fig_1_b_merfish, fig_1_b_tau = plot_fig_1_b(fig_1_b_merfish_data, fig_1_b_tau_data)

### Fig 1 C, L2/3 transcript enrichment analysis
### & significant fold change catagory fractions
### & Table S 1
fig_1_c_volcano, fig_1_c_categories, _, table_s_1 = plot_gene_enrichment(
    save_figs = False,
    meta_dict = meta_dict,
    used_merfish_genes_longnames = used_merfish_genes_longnames,
    paper_plotting = True,
    cross_layers = [0],
)


### Fig 1 D, R^2 of L2/3, L6 transcript -> tau models ###
fig_1_d_R2_by_layer, _, _, _, _, _, _ = plot_regressions(
    save_figs = False,
    generate_summary = True,
    meta_dict = meta_dict,
    used_merfish_genes_longnames = used_merfish_genes_longnames,
    regressions_to_plot = [1],
    paper_plotting = True,
    cross_layers = [0, 2],
)


### Fig 1 E, L2/3 transcript -> tau model betas 
### & beta categories
### & Table S 2
_, fig_1_e_model_betas, fig_1_e_pie_sig_beta_categories, fig_1_e_pie_sig_pos_beta_categories, table_s_2, _, _ = plot_regressions(
    save_figs = False,
    generate_summary = True,
    meta_dict = meta_dict,
    used_merfish_genes_longnames = used_merfish_genes_longnames,
    regressions_to_plot = [1],
    paper_plotting = True,
    cross_layers = [0],
)


### Fig S 1 A, L2/3 transcript & AP CCF & ML CCF -> tau model betas
### & beta categories
_, fig_s_1_a_XCCFmodel_betas, fig_s_1_a_XCCFmodel_pie_sig_beta_categories, _, _, _, _ = plot_regressions(
    save_figs = False,
    generate_summary = True,
    meta_dict = meta_dict,
    used_merfish_genes_longnames = used_merfish_genes_longnames,
    regressions_to_plot = [3],
    paper_plotting = True,
    cross_layers = [0],
)


### Fig S 1 B, tau vs AP CCF L2/3 model betas
### & beta categories
_, _, _, _, _, fig_s_1_b_tau_vs_AP_beta_fig, fig_a_1_b_tau_vs_AP_beta_category_fig = plot_regressions(
    save_figs = False,
    generate_summary = True,
    meta_dict = meta_dict,
    used_merfish_genes_longnames = used_merfish_genes_longnames,
    regressions_to_plot = [0],
    paper_plotting = True,
    cross_layers = [0],
)