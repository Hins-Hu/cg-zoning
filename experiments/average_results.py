"""Average the total demand served over repeated experiment runs.

Scans every run_* subdirectory of the given output folder, reads each
total_demand_served.txt, and writes the mean to average_demand_served.txt.
Used for the multi-run averages in Figure 3 (5 runs) and the equity study.
Usage: python experiments/average_results.py [output_dir]
"""

import os
import sys

# Directory containing the run_* output directories. Defaults to the repo's
# output/ folder; pass a different directory as the first CLI argument.
OUTPUT_BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "output"

# Get all run directories
run_dirs = [
    os.path.join(OUTPUT_BASE_DIR, d) for d in os.listdir(OUTPUT_BASE_DIR)
    if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d)) and d.startswith("run_")
]

if not run_dirs:
    print("No run directories found.")
    sys.exit(1)

# Calculate the average of the total demand served
total_demand = 0
num_runs = 0

for run_dir in run_dirs:
    demand_file = os.path.join(run_dir, "total_demand_served.txt")
    if os.path.isfile(demand_file):
        with open(demand_file, "r") as f:
            try:
                demand = float(f.read().strip())
                total_demand += demand
                num_runs += 1
            except ValueError:
                print(f"Warning: Invalid value in {demand_file}")
    else:
        print(f"Warning: {demand_file} not found.")

if num_runs == 0:
    print("No valid demand data found.")
    sys.exit(1)

average_demand = total_demand / num_runs
print(f"Average total demand served: {average_demand}")

# Save the average total demand served to a file
average_file = os.path.join(OUTPUT_BASE_DIR, "average_demand_served.txt")
with open(average_file, "w") as f:
    f.write(f"{average_demand}\n")

print(f"Average total demand served saved to: {average_file}")