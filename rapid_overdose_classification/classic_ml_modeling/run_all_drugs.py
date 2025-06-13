import subprocess
import sys
from rapid_overdose_classification.constants import all_drug_cols


if __name__ == "__main__":

    embedder = sys.argv[1]

    for drug in all_drug_cols:
        command = ["python", embedder, drug]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
