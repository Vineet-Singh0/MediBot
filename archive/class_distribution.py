import pandas as pd

# Load the dataset
df = pd.read_csv('data/new_p.csv')

# Print all column names to identify the label column
print('Columns in dataset:')
print(df.columns.tolist())

# Print the number of samples for each disease label (use 'disease' column)
print('Class distribution:')
print(df['disease'].value_counts())

# Optionally, print the total number of unique diseases
print(f'\nTotal unique diseases: {df["disease"].nunique()}')
