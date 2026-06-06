from scipy.stats import ttest_ind_from_stats

p_threshold = 0.05
equal_var=False

population = 5

def pop_std_to_sample_std(pop_std):
    return pop_std * (population / (population - 1)) ** 0.5

# Welsh's independent t test
# tuev: name, (kappa mean std), (bacc mean std), (wf1 mean std)
# chbmit: name, (bacc mean std), (auc-pr mean std), (auroc mean std)
my_model_metrics_tuev = ((74.23, 1.36), (75.57, 1.47), (86.88, 0.66))

their_model_name_tuev = (
"SPaRCNet",
"ContraWR",
"FFCL",
"CNN-Trans",
"ST-Trans",
"BIOT",
"LaBraM-Base",
"LaBraM-Large",
"LaBraM-Huge",
"CBraMod",
"Gram-B",
"Gram-M",
"Gram-L",
"MMM",
"BENDR",
"EEGPT-Tiny",
"EEGPT",
"NeuroLM-B",
"NeuroLM-L",
"NeuroLM-XL",
"REVE-base",
"NeurIPT",
"CSBrain",
"CodeBrain",
"Uni-NTFM-Large",
)
their_model_metrics_tuev = (
((42.33, 1.81), (41.61, 2.62), (70.24, 1.04)),
((39.12, 2.37), (43.84, 3.49), (68.93, 1.36)),
((37.32, 1.88), (39.79, 1.04), (67.83, 1.20)),
((38.15, 1.34), (40.87, 1.61), (68.54, 2.93)),
((37.65, 3.06), (39.84, 2.28), (68.23, 1.90)),
((52.73, 2.49), (52.81, 2.25), (74.92, 0.82)),
((64.33, 0.87), (67.25, 1.02), (82.01, 0.87)),
((66.22, 1.36), (65.81, 1.56), (83.15, 0.40)),
((66.80, 1.85), (66.95, 1.54), (84.01, 0.38)),
((67.72, 0.96), (66.71, 1.07), (83.42, 0.64)),
((65.28, 1.79), (73.26, 0.93), (86.14, 0.80)),
((66.77, 2.27), (74.06, 1.26), (86.74, 1.17)),
((71.30, 3.33), (74.87, 1.51), (88.24, 1.58)),
((23.22, 1.55), (52.69, 0.71), (68.20, 1.93)),
((39.64, 1.48), (50.09, 2.24), (75.35, 1.07)),
((50.85, 1.73), (56.70, 0.66), (75.35, 0.97)),
((63.51, 1.34), (62.32, 1.14), (81.87, 0.63)),
((42.85, 0.48), (45.60, 0.48), (71.53, 0.28)),
((44.14, 9.96), (41.32, 12.35), (73.87, 4.00)),
((45.70, 4.98), (46.79, 3.56), (73.59, 2.19)),
((67.83, 1.99), (67.59, 2.29), (84.51, 1.29)),
((69.70, 1.85), (67.61, 1.33), (84.28, 0.89)),
((68.33, 0.47), (69.03, 0.59), (83.33, 0.57)),
((69.12, 1.01), (64.28, 0.62), (83.62, 0.48)),
((70.30, 1.48), (69.91, 1.70), (84.66, 1.32)),
)

for name, (kappa, bacc, wf1) in zip(their_model_name_tuev, their_model_metrics_tuev):
    kappa_t, kappa_p = ttest_ind_from_stats(
        my_model_metrics_tuev[0][0], pop_std_to_sample_std(my_model_metrics_tuev[0][1]), population,
        kappa[0], pop_std_to_sample_std(kappa[1]), population,
        equal_var=equal_var, alternative="greater")
    
    bacc_t, bacc_p = ttest_ind_from_stats(
        my_model_metrics_tuev[1][0], pop_std_to_sample_std(my_model_metrics_tuev[1][1]), population,
        bacc[0], pop_std_to_sample_std(bacc[1]), population,
        equal_var=equal_var, alternative="greater")
    
    wf1_t, wf1_p = ttest_ind_from_stats(
        my_model_metrics_tuev[2][0], pop_std_to_sample_std(my_model_metrics_tuev[2][1]), population,
        wf1[0], pop_std_to_sample_std(wf1[1]), population,
        equal_var=equal_var, alternative="greater")
    # print(f"{name:>16}", kappa_p < p_threshold, bacc_p < p_threshold, wf1_p < p_threshold, sep="\t")
    print(f"{name:>16}", f"{kappa_p:.4f}", f"{bacc_p:.4f}", f"{wf1_p:.4f}", sep="\t")

my_model_metric_chbmit = ((85.82, 1.04), (48.45, 0.48), (89.26, 0.86)),