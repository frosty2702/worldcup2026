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

# target
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] < row["away_score"]:
        return 0
    else:
        return 1

matches["result"] = matches.apply(get_result, axis=1)

# -------------------------
# ADD FIFA RANKINGS
# -------------------------

# make dictionary
rank_dict = dict(zip(rankings["team"], rankings["rank"]))
points_dict = dict(zip(rankings["team"], rankings["points"]))

# add columns
matches["home_rank"] = matches["home_team"].map(rank_dict)
matches["away_rank"] = matches["away_team"].map(rank_dict)

matches["home_points"] = matches["home_team"].map(points_dict)
matches["away_points"] = matches["away_team"].map(points_dict)

# fill missing teams (older countries, historical teams)
matches["home_rank"] = matches["home_rank"].fillna(100)
matches["away_rank"] = matches["away_rank"].fillna(100)

matches["home_points"] = matches["home_points"].fillna(1000)
matches["away_points"] = matches["away_points"].fillna(1000)

# difference features
matches["rank_diff"] = matches["home_rank"] - matches["away_rank"]
matches["points_diff"] = matches["home_points"] - matches["away_points"]

# -------------------------
# ENCODE TEXT
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
        "rank_diff",
        "points_diff"
    ]
]

y = matches["result"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Better Model Accuracy:", accuracy)