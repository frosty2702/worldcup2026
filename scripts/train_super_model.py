import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# ============================================
# LOAD & ENGINEER FEATURES
# ============================================
df = pd.read_csv("data/final_dataset_v2.csv")

# Sort by year to compute temporal features
df = df.sort_values(['Year']).reset_index(drop=True)

# ============================================
# ADVANCED FEATURE ENGINEERING
# ============================================

# 1. RANKING-BASED FEATURES
df['rank_diff'] = df['home_rank'] - df['away_rank']
df['abs_rank_diff'] = df['rank_diff'].abs()
df['points_diff'] = df['home_points'] - df['away_points']
df['abs_points_diff'] = df['points_diff'].abs()
df['home_rank_strength'] = 1 / (df['home_rank'] + 1)  # Normalized rank strength
df['away_rank_strength'] = 1 / (df['away_rank'] + 1)

# 2. PERFORMANCE METRICS
df['goals_diff'] = df['home_avg_goals'] - df['away_avg_goals']
df['conceded_diff'] = df['home_avg_conceded'] - df['away_avg_conceded']
df['win_rate_diff'] = df['home_win_rate'] - df['away_win_rate']

# 3. EFFICIENCY & MOMENTUM
df['home_efficiency'] = df['home_avg_goals'] / (df['home_avg_goals'] + df['home_avg_conceded'] + 0.1)
df['away_efficiency'] = df['away_avg_goals'] / (df['away_avg_goals'] + df['away_avg_conceded'] + 0.1)
df['efficiency_diff'] = df['home_efficiency'] - df['away_efficiency']

# 4. COMPOSITE STRENGTH METRICS
df['home_strength'] = (df['home_rank_strength'] * 0.4 + 
                       df['home_win_rate'] * 0.3 + 
                       df['home_efficiency'] * 0.3)
df['away_strength'] = (df['away_rank_strength'] * 0.4 + 
                       df['away_win_rate'] * 0.3 + 
                       df['away_efficiency'] * 0.3)
df['strength_diff'] = df['home_strength'] - df['away_strength']

# 5. RANKING MOMENTUM (rolling statistics for teams)
team_momentum = {}
for team in df['home_team'].unique():
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if len(team_matches) > 0:
        avg_rank = team_matches[['home_rank', 'away_rank']].values.flatten().mean()
        team_momentum[team] = {'rank': avg_rank}

df['home_momentum'] = df['home_team'].map(lambda x: 1 / (team_momentum.get(x, {}).get('rank', 100) + 1))
df['away_momentum'] = df['away_team'].map(lambda x: 1 / (team_momentum.get(x, {}).get('rank', 100) + 1))
df['momentum_diff'] = df['home_momentum'] - df['away_momentum']

# 6. POLYNOMIAL FEATURES (interaction terms)
df['rank_ratio'] = (df['home_rank'] + 1) / (df['away_rank'] + 1)
df['points_ratio'] = (df['home_points'] + 1) / (df['away_points'] + 1)
df['goals_efficiency'] = df['home_avg_goals'] - df['away_avg_conceded']

# One-hot encode teams
df = pd.get_dummies(df, columns=["home_team", "away_team"], drop_first=False)

# Prepare features
X = df.drop(columns=["result", "Round", "home_score", "away_score", "Year"], errors="ignore")
X = X.select_dtypes(include=["int64", "float64", "bool"])
y = df["result"]

# Handle any NaN values
X = X.fillna(X.mean())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

print(f"📊 Dataset: {X.shape[0]} matches | {X.shape[1]} features")
print(f"📈 Class distribution: {dict(y.value_counts().sort_index())}")

# ============================================
# TRAIN/TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🔄 Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================
# LEVEL 0: BASE LEARNERS
# ============================================
print("\n⚙️  Training base learners...")

# LightGBM (very fast & accurate)
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# XGBoost with optimized params
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    objective='multi:softprob'
)

# Calibrated Random Forest
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=13,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_calibrated = CalibratedClassifierCV(rf_model, method='sigmoid', cv=5)

# Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)

# AdaBoost
ada_model = AdaBoostClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)

# Train all base learners
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train, verbose=False)
rf_calibrated.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
ada_model.fit(X_train, y_train)

# ============================================
# LEVEL 1: STACKING META-LEARNER
# ============================================
print("🏗️  Building meta-learner...")

# Generate meta-features from base learners
X_train_meta = np.column_stack([
    lgb_model.predict_proba(X_train),
    xgb_model.predict_proba(X_train),
    rf_calibrated.predict_proba(X_train),
    gb_model.predict_proba(X_train),
    ada_model.predict_proba(X_train)
])

X_test_meta = np.column_stack([
    lgb_model.predict_proba(X_test),
    xgb_model.predict_proba(X_test),
    rf_calibrated.predict_proba(X_test),
    gb_model.predict_proba(X_test),
    ada_model.predict_proba(X_test)
])

# Meta-learner: Logistic Regression
meta_learner = LogisticRegression(max_iter=1000, random_state=42)
meta_learner.fit(X_train_meta, y_train)

# ============================================
# PREDICTIONS & EVALUATION
# ============================================
print("\n" + "="*70)

lgb_pred = lgb_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
rf_pred = rf_calibrated.predict(X_test)
gb_pred = gb_model.predict(X_test)
ada_pred = ada_model.predict(X_test)
meta_pred = meta_learner.predict(X_test_meta)

# Calculate accuracies
lgb_acc = accuracy_score(y_test, lgb_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
rf_acc = accuracy_score(y_test, rf_pred)
gb_acc = accuracy_score(y_test, gb_pred)
ada_acc = accuracy_score(y_test, ada_pred)
meta_acc = accuracy_score(y_test, meta_pred)

print(f"📊 BASE LEARNER PERFORMANCE:")
print(f"  LightGBM:        {lgb_acc:.2%}")
print(f"  XGBoost:         {xgb_acc:.2%}")
print(f"  Random Forest:   {rf_acc:.2%}")
print(f"  Gradient Boost:  {gb_acc:.2%}")
print(f"  AdaBoost:        {ada_acc:.2%}")
print(f"\n  🏆 STACKING META: {meta_acc:.2%}")
print("="*70)

# Use best model
best_pred = meta_pred
best_acc = meta_acc

f1_weighted = f1_score(y_test, best_pred, average='weighted')
f1_macro = f1_score(y_test, best_pred, average='macro')

print(f"\n✨ FINAL METRICS:")
print(f"  Accuracy: {best_acc:.4f}")
print(f"  Weighted F1: {f1_weighted:.4f}")
print(f"  Macro F1: {f1_macro:.4f}")

print("\n" + "="*70)
print("CLASSIFICATION REPORT:")
print("="*70)
print(classification_report(y_test, best_pred, target_names=['Away Win', 'Draw', 'Home Win']))

print("\nCONFUSION MATRIX:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

print("\n🎯 PER-CLASS PERFORMANCE:")
for i, class_name in enumerate(['Away Win', 'Draw', 'Home Win']):
    if cm[i].sum() > 0:
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recall = cm[i, i] / cm[i].sum()
        print(f"  {class_name} ({['0', '1', '2'][i]})")
        print(f"    ├─ Recall: {recall:.2%}")
        print(f"    └─ Precision: {precision:.2%}")

# Feature importance from LightGBM
print("\n" + "="*70)
print("TOP 15 IMPORTANT FEATURES (LightGBM):")
print("="*70)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)
for idx, row in importance_df.iterrows():
    print(f"  {row['importance']:6.4f} ← {row['feature']}")

print("\n" + "="*70)
print(f"✅ Model training complete! Best accuracy: {best_acc:.2%}")
print("="*70)
