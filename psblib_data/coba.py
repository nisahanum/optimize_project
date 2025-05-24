import os

# Step 1: Define the parser
import re

import re

import re

def parse_sm_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    # === Extract number of jobs ===
    match = re.search(r'jobs \(incl\. supersource/sink \):\s+(\d+)', content)
    if match:
        n_jobs = int(match.group(1))
    else:
        n_jobs = 0

    jobs = {}

    # === Extract PRECEDENCE RELATIONS ===
    precedence_block = re.search(r'PRECEDENCE RELATIONS:(.*?)REQUESTS/DURATIONS:', content, re.DOTALL)
    if precedence_block:
        lines = precedence_block.group(1).splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("jobnr.") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                job_id = int(parts[0])
                n_successors = int(parts[2])
                successors = list(map(int, parts[3:3 + n_successors]))
                jobs[job_id] = {'successors': successors}

    # === Extract DURATIONS from REQUESTS/DURATIONS section ===
    durations_block = re.search(r'REQUESTS/DURATIONS:(.*?)RESOURCEAVAILABILITIES:', content, re.DOTALL)
    if durations_block:
        lines = durations_block.group(1).splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("jobnr.") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                job_id = int(parts[0])
                duration = int(parts[2])
                if job_id in jobs:
                    jobs[job_id]['duration'] = duration
                else:
                    jobs[job_id] = {'duration': duration, 'successors': []}

    return {
        'n_jobs': n_jobs,
        'jobs': jobs
    }





# Step 2: Set the folder path
folder_path = "/Users/nisahanum/Documents/S3/Bimbingan/SK 2/data set/j30.sm"

# Step 3: Loop through .sm files and parse them
files = [f for f in os.listdir(folder_path) if f.endswith('.sm')]

for fname in files:
    full_path = os.path.join(folder_path, fname)
    parsed = parse_sm_file(full_path)
    print(f"Parsed {fname}:")
    print(f"  Total Jobs: {parsed['n_jobs']}")
    print(f"  First 3 Jobs: {list(parsed['jobs'].items())[:3]}")
