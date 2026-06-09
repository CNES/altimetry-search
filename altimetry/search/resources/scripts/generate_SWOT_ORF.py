from swot_calval.io.indexer import calculate_orbit_repetition
from swot_calval.io import open_collection

import dask_jobqueue
import dask
from distributed import Client
import os

import numpy as np
import json

cluster = dask_jobqueue.SLURMCluster(
    cores=1,
    memory="8GiB", # Balanced configuration should have 8 GiB per core if possible
    processes=1,
    walltime="00:30:00",
    interface="ib0",
    # Please fill your account, obtained  with the command $myaccounts
    account="swotce_guest",
    job_extra_directives=["--export=None"], # Do not propagate environment
)
cluster.scale(jobs=5)
client = Client(cluster)

client.wait_for_workers(5)

db = open_collection("/work/HELPDESK_SWOTLR/commun/data/swot/L3_LR_SSH/zcollections/V3.0_CALVAL")
ds = db.query(cycle_numbers=range(474, 579), selected_variables=('time', 'cycle_number', 'pass_number'))

# db = open_collection("/work/HELPDESK_SWOTLR/commun/data/swot/L3_LR_SSH/zcollections/V3.0_SCIENCE")
# ds = db.query(cycle_numbers=range(1, 52), selected_variables=('time', 'cycle_number', 'pass_number'))

orf = calculate_orbit_repetition(ds)

# First pass for each cycle
_, first_indices = np.unique(orf['cycle_number'], return_index=True)
first_passes = orf[first_indices]

# Keep only if first pass is 1
first_passes = first_passes[first_passes['pass_number'] == 1]

# Build the dict
result = {
    str(int(row['cycle_number'])): str(row['start_time'])
    for row in first_passes
}

# Export to Json
with open('SWOT_science_ORF.json', 'w') as f:
    json.dump(result, f, indent=2)