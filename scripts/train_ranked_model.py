import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# load data
df = pd.read_csv("data/final_dataset.csv")

# encode teams
le = LabelEncoder()
df["home_team"] = le.fit_transform(df["home_team"])
df["away_team"] = le.fit_transform(df["away_team"])

# features
X = df[
    [
        "home_team",
        "away_team",
        "home_rank",
        "away_rank",
        "home_points",
        "away_points",
        "Year"
    ]
]

# target
y = df["result"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
# model
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Ranked Model Accuracy:", accuracy)