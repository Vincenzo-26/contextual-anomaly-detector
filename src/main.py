import json
from settings import PROJECT_ROOT
from find_groups_and_tw import run_groups_and_tw
from cmp_calculation import run_cmp
from prepare_case_study_data import prepare_case_study_data, find_parents_of_leaves
import os

case_study = "Cabina"
with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
    config = json.load(f)

sottocarichi = find_parents_of_leaves(config["Load Tree"])

prepare_case_study_data(case_study, sottocarichi)
run_groups_and_tw(case_study)

run_cmp(case_study)

# anomaly table dei sottocarichi basato su groups e tw di Cabina per estrarre frequenze per CPD della BN
for sottocarico in sottocarichi:
    run_cmp(sottocarico, case_study)



