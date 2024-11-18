import subprocess
import sys

all_drug_cols = [
    "Methamphetamine",
    "Heroin",
    "Cocaine",
    "Fentanyl",
    "Alcohol",
    "Prescription.opioids",
    "Any Opioids",
    "Benzodiazepines",
    "Others",
    "Any Drugs",
    "Drug No Opioids",
]

if __name__ == "__main__":

    embedder = sys.argv[1]

    for drug in all_drug_cols:
        command = ["python", embedder, drug]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
