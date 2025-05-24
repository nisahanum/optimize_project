import os
import re
import json
import pandas as pd

# === CONFIGURATION ===
base_dir = r"C:\Users\nisahanum\Documents\cobagit\optimize_project\psblib_data\data\j30.mm"
csv_path = os.path.join(base_dir, "ifpom_projects_50_60.csv")
json_path = os.path.join(base_dir, "ifpom_projects_50_60.json")

# === PARSER FUNCTION ===
def parse_mm_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    jobs = {}
    reading_successors = False
    reading_requests = False

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip empty or decorative lines
        if not line or line.startswith('*') or line.startswith('-') or 'jobnr.' in line.lower():
            continue

        if line.startswith("PRECEDENCE RELATIONS"):
            reading_successors = True
            continue
        elif line.startswith("REQUESTS/DURATIONS"):
            reading_successors = False
            reading_requests = True
            continue
        elif line.startswith("RESOURCEAVAILABILITIES"):
            reading_requests = False
            continue

        # Process successors
        if reading_successors:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                job_id = int(parts[0])
                num_successors = int(parts[2])
                successors = list(map(int, parts[3:3 + num_successors]))
                jobs[job_id] = {'successors': successors, 'modes': {}}

        # Process durations and modes
        if reading_requests:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                job_id = int(parts[0])
                mode = int(parts[1])
                duration = int(parts[2])
                resources = list(map(int, parts[3:]))

                if job_id not in jobs:
                    jobs[job_id] = {'successors': [], 'modes': {}}
                jobs[job_id]['modes'][mode] = {
                    'duration': duration,
                    'resources': resources
                }

    return jobs


# === GET ALL .mm FILES ===
files = sorted(f for f in os.listdir(base_dir) if f.endswith('.mm'))
selected_files = files[49:60]  # files 50 to 60 (index 49 to 59)

parsed_projects = {}

for fname in selected_files:
    full_path = os.path.join(base_dir, fname)
    project_id = fname.replace('.mm', '')
    parsed_projects[project_id] = parse_mm_file(full_path)

# === CONVERT TO FLATTENED DATAFRAME FOR CSV ===
rows = []
for project_id, jobs in parsed_projects.items():
    for job_id, job_data in jobs.items():
        for mode_id, mode_data in job_data['modes'].items():
            row = {
                'project_id': project_id,
                'job_id': job_id,
                'mode_id': mode_id,
                'duration': mode_data['duration'],
                'resources': mode_data['resources'],
                'successors': job_data['successors']
            }
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)
print(f"CSV exported to: {csv_path}")

# === EXPORT RAW STRUCTURE TO JSON ===
with open(json_path, 'w') as jf:
    json.dump(parsed_projects, jf, indent=2)
print(f"JSON exported to: {json_path}")

