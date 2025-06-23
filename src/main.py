import time
from datetime import datetime
import os
os.environ["OMP_NUM_THREADS"] = "1"
from find_groups_and_tw import run_groups_and_tw
from cmp_calculation import run_cmp
from prepare_case_study_data import run_data
from bayesian_network import run_BN
from calc_energy_anm import run_soft_evd_LR
from utils import *
from calc_daily_thermal_sens import find_thermal_sens
from calc_soft_evidence import combine_soft_evidence
from calc_temp_anm import calc_temp_anm_prob

start_time = time.time()
start_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Starting analysis: {start_readable}")

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

# CMP
run_cmp(case_study)

# Creazione energy evidences
run_soft_evd_LR(case_study, 50, 0.6, 0.8)

# Creazione thermal evidences
find_thermal_sens(case_study, 1500, "minmax")
calc_temp_anm_prob(case_study)

# Combinazione delle energy evidences e thermal evidences
combine_soft_evidence(case_study)

# Creazione rete bayesiana e inferenza
inference_results = run_BN(case_study)

end_time = time.time()
end_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Analisys finished: {end_readable}")
minutes, seconds = divmod((end_time - start_time), 60)
print(f"Execution time: {int(minutes)} min {int(seconds)} sec")



