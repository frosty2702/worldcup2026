import pandas as pd

# load files
matches = pd.read_csv("data/final_dataset.csv")
stats = pd.read_csv("data/team_stats.csv")

# reset index if team names are index
stats = stats.rename(columns={"Unnamed: 0": "team"})

# HOME TEAM merge
matches = matches.merge(
    stats[["team", "win_rate", "avg_goals", "avg_conceded"]],
    left_on="home_team",
    right_on="team",
    how="left"
)

matches = matches.rename(columns={
    "win_rate": "home_win_rate",
    "avg_goals": "home_avg_goals",
    "avg_conceded": "home_avg_conceded"
})

matches = matches.drop(columns=["team"])

# AWAY TEAM merge
matches = matches.merge(
    stats[["team", "win_rate", "avg_goals", "avg_conceded"]],
    left_on="away_team",
    right_on="team",
    how="left"
)

matches = matches.rename(columns={
    "win_rate": "away_win_rate",
    "avg_goals": "away_avg_goals",
    "avg_conceded": "away_avg_conceded"
})

matches = matches.drop(columns=["team"])

print(matches.head())

matches.to_csv("data/final_dataset_v2.csv", index=False)

print("Saved final_dataset_v2.csv")