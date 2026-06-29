import pandas as pd

schedule = pd.read_csv("data/raw/schedule_2026.csv")

print(schedule.head(20))

print("\nColumns:")
print(schedule.columns)