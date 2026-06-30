import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
from pathlib import Path


def build_features(df):
    # user-suggested differences
    df = df.copy()
    df['rank_diff'] = df['away_rank'] - df['home_rank']
    df['points_diff'] = df['home_points'] - df['away_points']
    df['winrate_diff'] = df['home_win_rate'] - df['away_win_rate']
    df['goal_diff'] = df['home_avg_goals'] - df['away_avg_goals']
    df['abs_rank_diff'] = df['rank_diff'].abs()
    df['abs_points_diff'] = df['points_diff'].abs()
    return df[
        [
            'rank_diff', 'points_diff', 'winrate_diff', 'goal_diff',
            'abs_rank_diff', 'abs_points_diff'
        ]
    ]


def main():
    df = pd.read_csv('data/final_dataset_v2.csv')

    X = build_features(df)
    y = df['result']

    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Weighted F1:', f1_score(y_test, y_pred, average='weighted'))
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=['Away Win','Draw','Home Win']))

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / 'diff_model.joblib')
    joblib.dump(scaler, models_dir / 'diff_scaler.joblib')

    print(f"Saved model to {models_dir / 'diff_model.joblib'} and scaler to {models_dir / 'diff_scaler.joblib'}")


if __name__ == '__main__':
    main()
