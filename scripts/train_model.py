import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load dataset
matches = pd.read_csv("data/raw/matches_1930_2022.csv")

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

# target column
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] < row["away_score"]:
        return 0
    else:
        return 1

matches["result"] = matches.apply(get_result, axis=1)

# encode teams
team_encoder = LabelEncoder()
all_teams = pd.concat([matches["home_team"], matches["away_team"]])
team_encoder.fit(all_teams)

matches["home_team"] = team_encoder.transform(matches["home_team"])
matches["away_team"] = team_encoder.transform(matches["away_team"])

# encode round
round_encoder = LabelEncoder()
matches["Round"] = round_encoder.fit_transform(matches["Round"])

# features (X)
X = matches[
    [
        "home_team",
        "away_team",
        "Round",
        "Year"
    ]
]

# target (y)
y = matches["result"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create model
model = RandomForestClassifier(n_estimators=100)

# train
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)