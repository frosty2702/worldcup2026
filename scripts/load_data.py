import pandas as pd

# load historical matches
matches = pd.read_csv("data/raw/matches_1930_2022.csv")

# basic check
print("Dataset loaded successfully")
print()

print("Number of matches:")
print(matches.shape)

print()

print("Columns:")
print(matches.columns)

print()

print("First 5 rows:")
print(matches.head())