import pandas as pd
from sklearn.preprocessing import LabelEncoder

# load data
matches = pd.read_csv("data/raw/matches_1930_2022.csv")

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

# create target column
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2      # home win
    elif row["home_score"] < row["away_score"]:
        return 0      # away win
    else:
        return 1      # draw

matches["result"] = matches.apply(get_result, axis=1)

# encode team names
team_encoder = LabelEncoder()

all_teams = pd.concat(
    [matches["home_team"], matches["away_team"]]
)

team_encoder.fit(all_teams)

matches["home_team"] = team_encoder.transform(matches["home_team"])
matches["away_team"] = team_encoder.transform(matches["away_team"])

# encode tournament round
round_encoder = LabelEncoder()

matches["Round"] = round_encoder.fit_transform(matches["Round"])

print(matches.head(10))

print("\nExample team mapping:")
for i in range(10):
    print(i, "=", team_encoder.inverse_transform([i])[0])