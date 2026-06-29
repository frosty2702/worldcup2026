import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# LOAD TRAINING DATA
# -------------------------
matches = pd.read_csv("data/raw/matches_1930_2022.csv")
rankings = pd.read_csv("data/raw/fifa_ranking_2022-10-06.csv")
schedule = pd.read_csv("data/raw/schedule_2026.csv")

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

# target
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] < row["away_score"]:
        return 0
    return 1

matches["result"] = matches.apply(get_result, axis=1)

# -------------------------
# TEAM STATS
# -------------------------
teams = {}

for _, row in matches.iterrows():

    h = row["home_team"]
    a = row["away_team"]

    if h not in teams:
        teams[h] = {"games":0,"goals":0,"wins":0}

    if a not in teams:
        teams[a] = {"games":0,"goals":0,"wins":0}

    teams[h]["games"] += 1
    teams[h]["goals"] += row["home_score"]

    teams[a]["games"] += 1
    teams[a]["goals"] += row["away_score"]

    if row["home_score"] > row["away_score"]:
        teams[h]["wins"] += 1
    elif row["away_score"] > row["home_score"]:
        teams[a]["wins"] += 1

stats = {}

for team, data in teams.items():

    stats[team] = {
        "avg_goals": data["goals"] / data["games"],
        "win_rate": data["wins"] / data["games"]
    }

# -------------------------
# FIFA RANKS
# -------------------------
rank_dict = dict(zip(rankings["team"], rankings["rank"]))

matches["home_rank"] = matches["home_team"].map(rank_dict).fillna(100)
matches["away_rank"] = matches["away_team"].map(rank_dict).fillna(100)

matches["home_avg_goals"] = matches["home_team"].map(
    lambda x: stats[x]["avg_goals"]
)

matches["away_avg_goals"] = matches["away_team"].map(
    lambda x: stats[x]["avg_goals"]
)

matches["home_win_rate"] = matches["home_team"].map(
    lambda x: stats[x]["win_rate"]
)

matches["away_win_rate"] = matches["away_team"].map(
    lambda x: stats[x]["win_rate"]
)

# -------------------------
# ENCODERS
# -------------------------
team_encoder = LabelEncoder()

all_teams = pd.concat(
    [matches["home_team"], matches["away_team"]]
)

team_encoder.fit(all_teams)

matches["home_team"] = team_encoder.transform(matches["home_team"])
matches["away_team"] = team_encoder.transform(matches["away_team"])

round_encoder = LabelEncoder()
matches["Round"] = round_encoder.fit_transform(matches["Round"])

# training data
X = matches[
    [
        "home_team",
        "away_team",
        "Round",
        "Year",
        "home_rank",
        "away_rank",
        "home_avg_goals",
        "away_avg_goals",
        "home_win_rate",
        "away_win_rate"
    ]
]

y = matches["result"]

# train model
model = RandomForestClassifier(n_estimators=300)
model.fit(X, y)

# -------------------------
# PREPARE 2026 MATCHES
# -------------------------

# keep columns
schedule = schedule[
    [
        "home_team",
        "away_team",
        "Round",
        "Year"
    ]
]

# encode teams safely
def safe_team(team):
    if team in team_encoder.classes_:
        return team_encoder.transform([team])[0]
    return -1

schedule["home_rank"] = schedule["home_team"].map(rank_dict).fillna(100)
schedule["away_rank"] = schedule["away_team"].map(rank_dict).fillna(100)

schedule["home_avg_goals"] = schedule["home_team"].map(
    lambda x: stats[x]["avg_goals"] if x in stats else 1.0
)

schedule["away_avg_goals"] = schedule["away_team"].map(
    lambda x: stats[x]["avg_goals"] if x in stats else 1.0
)

schedule["home_win_rate"] = schedule["home_team"].map(
    lambda x: stats[x]["win_rate"] if x in stats else 0.3
)

schedule["away_win_rate"] = schedule["away_team"].map(
    lambda x: stats[x]["win_rate"] if x in stats else 0.3
)

schedule["home_team_encoded"] = schedule["home_team"].apply(safe_team)
schedule["away_team_encoded"] = schedule["away_team"].apply(safe_team)

schedule["Round"] = round_encoder.transform(schedule["Round"])

X_future = schedule[
    [
        "home_team_encoded",
        "away_team_encoded",
        "Round",
        "Year",
        "home_rank",
        "away_rank",
        "home_avg_goals",
        "away_avg_goals",
        "home_win_rate",
        "away_win_rate"
    ]
]

# rename columns to match training
X_future.columns = X.columns

# predict
predictions = model.predict(X_future)

# decode
result_map = {
    0: "Away Win",
    1: "Draw",
    2: "Home Win"
}

schedule["prediction"] = [
    result_map[p] for p in predictions
]

print(
    schedule[
        [
            "home_team",
            "away_team",
            "prediction"
        ]
    ].head(20)
)