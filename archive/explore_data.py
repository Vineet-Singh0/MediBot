import pandas as pd
import os

# Define data folder path
data_folder = 'data'

# List all CSV files in the data folder
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

print('Found CSV files:', csv_files)

# Load and display info for each CSV file
def explore_csv(file_path):
    print(f'\nExploring {file_path}')
    df = pd.read_csv(file_path)
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('First 5 rows:')
    print(df.head())
    print('Missing values per column:')
    print(df.isnull().sum())
    print('-' * 40)

for csv_file in csv_files:
    explore_csv(os.path.join(data_folder, csv_file))
