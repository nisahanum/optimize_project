import os
import re

# Update this to your local path
folder_path = r"C:\Users\nisahanum\Documents\cobagit\optimize_project\psblib_data\data\j30.mm"

def parse_mm_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    jobs = {}
    reading_precedence = False
    reading_modes = False

    for line in lines:
        line = line.strip()

        if line.startswith("PRECEDENCE RELATIONS:"):
            reading_precedence = True
            continue
        if line.startswith("REQUESTS/DURATIONS:"):
            reading_precedence = False
            reading_modes = True
            continue
        if line.startswith("RESOURCEAVAILABILITIES:"):
            reading_modes = False
            continue

        # === Skip empty lines and header rows ===
        if not line or any(x in line for x in ['jobnr', 'mode', '----']):
            continue

        # === Parse PRECEDENCE RELATIONS ===
        if reading_precedence:
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                job_id = int(parts[0])
                n_successors = int(parts[2])
                successors = list(map(int, parts[3:3 + n_successors]))
                jobs[job_id] = {'successors': successors, 'modes': {}}

        # === Parse REQUESTS/DURATIONS ===
        if reading_modes:
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit():
                job_id = int(parts[0])
                mode_id = int(parts[1])
                duration = int(parts[2])
                resources = list(map(int, parts[3:]))
                if job_id not in jobs:
                    jobs[job_id] = {'successors': [], 'modes': {}}
                jobs[job_id]['modes'][mode_id] = {'duration': duration, 'resources': resources}

    return jobs



# Read all files in the folder
all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mm')])
sample_files = all_files[:3]  # Just the first 3 for example

parsed_projects = {}

for fname in sample_files:
    full_path = os.path.join(folder_path, fname)
    project_id = fname.replace('.mm', '')
    parsed_projects[project_id] = parse_mm_file(full_path)
    print(f"Parsed {project_id}:")
    for job_id, job_data in list(parsed_projects[project_id].items())[:3]:
        print(f"  Job {job_id}:")
        print(f"    Successors: {job_data['successors']}")
        for mode_id, mode_info in job_data['modes'].items():
            print(f"      Mode {mode_id}: Duration={mode_info['duration']}, Resources={mode_info['resources']}")
        print("-" * 50)
