import pandas as pd

matches = pd.read_csv("data/cleaned_matches.csv")
rankings = pd.read_csv("data/raw/fifa_ranking_2026-06-08.csv")

rankings = rankings[["team", "rank", "points"]]

# home merge
matches = matches.merge(
    rankings,
    left_on="home_team",
    right_on="team",
    how="left"
)

matches = matches.rename(columns={
    "rank": "home_rank",
    "points": "home_points"
})

print("\nAfter home merge:")
print(matches.isnull().sum())

# teams not matched
print("\nUnmatched home teams:")
print(matches[matches["home_rank"].isnull()]["home_team"].unique())

matches = matches.drop(columns=["team"])

# away merge
matches = matches.merge(
    rankings,
    left_on="away_team",
    right_on="team",
    how="left"
)

matches = matches.rename(columns={
    "rank": "away_rank",
    "points": "away_points"
})

print("\nAfter away merge:")
print(matches.isnull().sum())

# teams not matched
print("\nUnmatched away teams:")
print(matches[matches["away_rank"].isnull()]["away_team"].unique())

matches = matches.drop(columns=["team"])
# after merge
# check missing values after both merges
print("\nBefore dropping NaN:")
print(matches.isnull().sum())

# remove rows where rankings were missing
matches = matches.dropna()

print("\nAfter dropping NaN:")
print(matches.isnull().sum())
matches.to_csv("data/final_dataset.csv", index=False)

print("\nSaved final_dataset.csv")