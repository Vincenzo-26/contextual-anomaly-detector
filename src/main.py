import os
os.environ["OMP_NUM_THREADS"] = "1"
from find_groups_and_tw import run_groups_and_tw
from cmp_calculation import run_cmp
from prepare_case_study_data import run_data
from bayesian_network import run_BN
from calc_energy_distr import run_soft_evd_EM
from utils import *
from change_point_detection import run_change_point
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

# CMP alto livello
run_cmp(case_study)

# Creazione energy evidences
run_soft_evd_EM(case_study)

# Creazione thermal evidences
run_change_point(case_study, penalty=10)

# Combinazione delle energy evidences e thermal evidences
combine_soft_evidence(case_study)

# Creazione rete bayesiana e inferenza
inference_results = run_BN(case_study)



