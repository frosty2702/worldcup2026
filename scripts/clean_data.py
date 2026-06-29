import pandas as pd

matches = pd.read_csv("data/raw/matches_1930_2022.csv")
# keep only modern era matches
matches = matches[matches["Year"] >= 2010]
# keep useful columns
matches = matches[
    [
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "Round",
        "Year"
    ]
]

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2          # home win
    elif row["home_score"] < row["away_score"]:
        return 0          # away win
    else:
        return 1          # draw

matches["result"] = matches.apply(get_result, axis=1)

print(matches.head())

print("\nMissing values:")
print(matches.isnull().sum())

# SAVE CLEANED DATA  ← this was missing
matches.to_csv("data/cleaned_matches.csv", index=False)

print("\nSaved cleaned_matches.csv successfully")