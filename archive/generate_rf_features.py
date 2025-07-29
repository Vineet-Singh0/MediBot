import pandas as pd

# Load the dataset
df = pd.read_csv('data/p.csv')

# Find the feature columns: from after 'age_group' to before 'gender'
columns = list(df.columns)
start = columns.index('age_group') + 1
end = columns.index('gender')
features = columns[start:end]

# Write features to rf_features.txt
with open('features/rf_features.txt', 'w') as f:
    for feat in features:
        f.write(feat + '\n')

print(f"Extracted {len(features)} features to rf_features.txt")
