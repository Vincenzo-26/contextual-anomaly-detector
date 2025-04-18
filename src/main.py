from find_groups_and_tw import run_groups_and_tw
from cmp_calculation import run_cmp
from prepare_case_study_data import run_data
from bayesian_network import build_BN_structural_model
from utils import *


case_study = "Cabina"
with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
    config = json.load(f)

sottocarichi = find_parents_of_leaves(config["Load Tree"])

# creazione dei dataframe (uno per ogni nodo foglia + uno aggregato per il livello a loro superiore)
run_data(case_study, sottocarichi)

# estrazione groups e tw ad alto livello
run_groups_and_tw(case_study)

# CMP alto livello
run_cmp(case_study)

# CMP ai nodi foglia utilizzando groups e tw estratte ad alto livello
for sottocarico in sottocarichi:
    run_cmp(sottocarico, case_study)

# Creazione rete bayesiana
model = build_BN_structural_model(case_study)



