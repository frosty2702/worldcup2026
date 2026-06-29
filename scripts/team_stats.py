import pandas as pd

matches = pd.read_csv("data/raw/matches_1930_2022.csv")

# keep only needed columns
matches = matches[
    [
        "home_team",
        "away_team",
        "home_score",
        "away_score"
    ]
]

teams = {}

for _, row in matches.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    # initialize
    if home not in teams:
        teams[home] = {
            "games": 0,
            "goals_scored": 0,
            "goals_conceded": 0,
            "wins": 0
        }

    if away not in teams:
        teams[away] = {
            "games": 0,
            "goals_scored": 0,
            "goals_conceded": 0,
            "wins": 0
        }

    # home stats
    teams[home]["games"] += 1
    teams[home]["goals_scored"] += row["home_score"]
    teams[home]["goals_conceded"] += row["away_score"]

    # away stats
    teams[away]["games"] += 1
    teams[away]["goals_scored"] += row["away_score"]
    teams[away]["goals_conceded"] += row["home_score"]

    # wins
    if row["home_score"] > row["away_score"]:
        teams[home]["wins"] += 1
    elif row["away_score"] > row["home_score"]:
        teams[away]["wins"] += 1

# convert to dataframe
stats = pd.DataFrame.from_dict(teams, orient="index")

stats["avg_goals"] = stats["goals_scored"] / stats["games"]
stats["avg_conceded"] = stats["goals_conceded"] / stats["games"]
stats["win_rate"] = stats["wins"] / stats["games"]

print(stats.head(20))
stats.to_csv("data/team_stats.csv")
print("Saved team_stats.csv")