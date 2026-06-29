import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


# load dataset
df = pd.read_csv("data/final_dataset_v2.csv")


# convert team names into numeric columns
df = pd.get_dummies(
    df,
    columns=["home_team", "away_team"]
)


# remove columns we should not train on
# result = target
# Round = text column (causes errors)
X = df.drop(
    columns=[
        "result",
        "Round"
    ],
    errors="ignore"
)


# keep only numeric columns automatically
X = X.select_dtypes(
    include=["int64", "float64", "bool"]
)


# target
y = df["result"]


# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# model
model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)


# train
model.fit(X_train, y_train)


# predict
pred = model.predict(X_test)


# accuracy
accuracy = accuracy_score(y_test, pred)


print("Super Model Accuracy:", accuracy)


# class balance
print("\nResult distribution:")
print(df["result"].value_counts())