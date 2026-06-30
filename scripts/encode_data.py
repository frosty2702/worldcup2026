import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

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

# add simple engineered features
matches["goal_diff"] = matches["home_score"] - matches["away_score"]
matches["total_goals"] = matches["home_score"] + matches["away_score"]

# create target column
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2      # home win
    elif row["home_score"] < row["away_score"]:
        return 0      # away win
    else:
        return 1      # draw

matches["result"] = matches.apply(get_result, axis=1)

# encode team names with a stable mapping and handle unseen teams as -1
team_encoder = LabelEncoder()

all_teams = pd.concat([matches["home_team"], matches["away_team"]]).unique()
team_encoder.fit(all_teams)

# build mapping dict from fitted classes_
team_mapping = {team: idx for idx, team in enumerate(team_encoder.classes_)}

matches["home_team"] = matches["home_team"].map(team_mapping).fillna(-1).astype(int)
matches["away_team"] = matches["away_team"].map(team_mapping).fillna(-1).astype(int)

# encode tournament round
round_encoder = LabelEncoder()
matches["Round"] = round_encoder.fit_transform(matches["Round"].astype(str))

# ensure models dir exists and save encoders
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(team_encoder, models_dir / "team_encoder.joblib")
joblib.dump(round_encoder, models_dir / "round_encoder.joblib")

# export cleaned/encoded dataset
out_path = Path("data/cleaned_matches.csv")
matches.to_csv(out_path, index=False)

print(matches.head(10))
print(f"\nSaved cleaned data to: {out_path}")
print(f"Saved encoders to: {models_dir / 'team_encoder.joblib'}, {models_dir / 'round_encoder.joblib'}")

print("\nExample team mapping (first 10 classes):")
for i, team in enumerate(team_encoder.classes_[:10]):
    print(i, "=", team)