import pandas as pd

rankings = pd.read_csv("data/raw/fifa_ranking_2022-10-06.csv")

print(rankings.head(20))

print("\nColumns:")
print(rankings.columns)