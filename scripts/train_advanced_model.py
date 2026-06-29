import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# LOAD DATA
# -------------------------
matches = pd.read_csv("data/raw/matches_1930_2022.csv")
rankings = pd.read_csv("data/raw/fifa_ranking_2022-10-06.csv")

# keep columns
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

# -------------------------
# TARGET
# -------------------------
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] < row["away_score"]:
        return 0
    return 1

matches["result"] = matches.apply(get_result, axis=1)

# -------------------------
# TEAM HISTORICAL STATS
# -------------------------
teams = {}

for _, row in matches.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    if home not in teams:
        teams[home] = {"games":0,"goals":0,"conceded":0,"wins":0}

    if away not in teams:
        teams[away] = {"games":0,"goals":0,"conceded":0,"wins":0}

    teams[home]["games"] += 1
    teams[home]["goals"] += row["home_score"]
    teams[home]["conceded"] += row["away_score"]

    teams[away]["games"] += 1
    teams[away]["goals"] += row["away_score"]
    teams[away]["conceded"] += row["home_score"]

    if row["home_score"] > row["away_score"]:
        teams[home]["wins"] += 1

    elif row["away_score"] > row["home_score"]:
        teams[away]["wins"] += 1

stats = {}

for team, data in teams.items():

    stats[team] = {
        "avg_goals": data["goals"] / data["games"],
        "avg_conceded": data["conceded"] / data["games"],
        "win_rate": data["wins"] / data["games"]
    }

# -------------------------
# ADD TEAM STATS
# -------------------------
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
# ADD FIFA RANKS
# -------------------------
rank_dict = dict(zip(rankings["team"], rankings["rank"]))

matches["home_rank"] = matches["home_team"].map(rank_dict)
matches["away_rank"] = matches["away_team"].map(rank_dict)

matches["home_rank"] = matches["home_rank"].fillna(100)
matches["away_rank"] = matches["away_rank"].fillna(100)

# -------------------------
# ENCODE
# -------------------------
encoder = LabelEncoder()

all_teams = pd.concat(
    [matches["home_team"], matches["away_team"]]
)

encoder.fit(all_teams)

matches["home_team"] = encoder.transform(matches["home_team"])
matches["away_team"] = encoder.transform(matches["away_team"])

round_encoder = LabelEncoder()
matches["Round"] = round_encoder.fit_transform(matches["Round"])

# -------------------------
# FEATURES
# -------------------------
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

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier(n_estimators=300)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Advanced Model Accuracy:", accuracy)