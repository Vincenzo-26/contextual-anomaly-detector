import os
os.environ["OMP_NUM_THREADS"] = "1"
from find_groups_and_tw import run_groups_and_tw
from cmp_calculation import run_cmp
from prepare_case_study_data import run_data
from bayesian_network import run_BN
from calc_energy_anm import run_soft_evd_LR
from utils import *
from calc_thermal_anm import run_change_point
from calc_soft_evidence import combine_soft_evidence


case_study = "Cabina"
with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
    config = json.load(f)

sottocarichi = find_parents_of_leaves(config["Load Tree"])

# creazione dei dataframe (uno per ogni nodo foglia + uno aggregato per il livello a loro superiore)
for sottocarico in sottocarichi:
    run_data(sottocarico)
run_data(case_study, sottocarichi)

# estrazione groups e tw ad alto livello
run_groups_and_tw(case_study)

# # CMP
# run_cmp(case_study)

# Creazione energy evidences
run_soft_evd_LR(case_study, 50, 0.6, 0.8)

# Creazione thermal evidences
run_change_point(case_study, penalty=100, bic_threshold= 0.2, norm_for_check_term_sens = False)

# Combinazione delle energy evidences e thermal evidences
combine_soft_evidence(case_study, 5)

# Creazione rete bayesiana e inferenza
inference_results = run_BN(case_study)



