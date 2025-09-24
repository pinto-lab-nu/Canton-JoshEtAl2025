import os
import pickle
repo_dir = r"\user\project_dir\expressionTauRegressions"
os.chdir(repo_dir)
from expression_script import *


# Regression type codes:
# 0 : transcript expression                     -> AP CCF & ML CCF
# 1 : transcript expression                     -> tau
# 2 : transcript expression                     -> tau residuals
# 3 : transcript expression & AP CCF & ML CCF   -> tau

# Cross layers codes (comparing model performance for merfish data isolated to different layers):
# 0 : L2/3
# 1 : L4/5
# 2 : L6

# Assumes regressions have already been run with -> run_transcript_tau_GLMs()

### Load Plotting Data to Make figures ###
glm_plotting_output_path = os.path.join(repo_dir, 'Plotting', 'Cux2-Ai96', 'pooling0.1mm', 'GenePredictors', 'Merfish-Imputed')
meta_dict = pickle.load(open(os.path.join(glm_plotting_output_path, f'plotting_data.pickle'), 'rb'))
used_merfish_genes_longnames = pickle.load(open(os.path.join(repo_dir, 'Plotting', f'merfishImputed_usedLongGeneNames.pickle'), 'rb'))


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