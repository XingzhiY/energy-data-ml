import pandas as pd

# Load data
df = pd.read_csv('owid-energy-data.csv')

# Display all column names
print("All columns in the dataset:")
print(df.columns.tolist())

# Display basic dataset information
print("\nBasic dataset information:")
print(df.info())