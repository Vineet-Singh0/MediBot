import os

# List of model and encoder files to keep (the best deduplicated-by-symptoms version)
files_to_keep = [
    'final_symptom_checker_model_dedup_symptoms.pkl',
    'label_encoder_dedup_symptoms.pkl'
]

# List all .pkl files in the current directory
all_pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]

# Delete unwanted .pkl files
for f in all_pkl_files:
    if f not in files_to_keep:
        print(f"Deleting: {f}")
        os.remove(f)
print("Cleanup complete. Only the best model and encoder remain.")
