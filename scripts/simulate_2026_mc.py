import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict, deque


def team_stats_from_matches(matches_path):
    matches = pd.read_csv(matches_path)
    teams = {}
    for _, row in matches.iterrows():
        h = row['home_team']
        a = row['away_team']
        for t in (h, a):
            if t not in teams:
                teams[t] = {'games': 0, 'goals': 0, 'wins': 0}
        teams[h]['games'] += 1
        teams[h]['goals'] += row.get('home_score', 0)
        teams[a]['games'] += 1
        teams[a]['goals'] += row.get('away_score', 0)
        if row.get('home_score', 0) > row.get('away_score', 0):
            teams[h]['wins'] += 1
        elif row.get('away_score', 0) > row.get('home_score', 0):
            teams[a]['wins'] += 1
    stats = {}
    for t, d in teams.items():
        stats[t] = {
            'avg_goals': d['goals'] / d['games'] if d['games'] > 0 else 1.0,
            'win_rate': d['wins'] / d['games'] if d['games'] > 0 else 0.3
        }
    return stats


def infer_groups(schedule_df):
    edges = defaultdict(set)
    for _, r in schedule_df[schedule_df['Round'].str.contains('Group', na=False)].iterrows():
        h = r['home_team']
        a = r['away_team']
        edges[h].add(a)
        edges[a].add(h)
    visited = set()
    groups = []
    for node in edges:
        if node in visited:
            continue
        comp = []
        dq = deque([node])
        visited.add(node)
        while dq:
            n = dq.popleft()
            comp.append(n)
            for nei in edges[n]:
                if nei not in visited:
                    visited.add(nei)
                    dq.append(nei)
        groups.append(sorted(comp))
    return groups


def build_match_features(row, rank_dict, rank_points, stats):
    home = row['home_team']
    away = row['away_team']
    home_rank = rank_dict.get(home, 100)
    away_rank = rank_dict.get(away, 100)
    home_points = rank_points.get(home, 0)
    away_points = rank_points.get(away, 0)
    home_win = stats.get(home, {}).get('win_rate', 0.3)
    away_win = stats.get(away, {}).get('win_rate', 0.3)
    home_avg_goals = stats.get(home, {}).get('avg_goals', 1.0)
    away_avg_goals = stats.get(away, {}).get('avg_goals', 1.0)

    rank_diff = away_rank - home_rank
    points_diff = home_points - away_points
    winrate_diff = home_win - away_win
    goal_diff = home_avg_goals - away_avg_goals
    abs_rank_diff = abs(rank_diff)
    abs_points_diff = abs(points_diff)

    return [rank_diff, points_diff, winrate_diff, goal_diff, abs_rank_diff, abs_points_diff]


def simulate_once(model, scaler, schedule, rank_dict, rank_points, stats, rng):
    # predict group matches
    feats = []
    for _, r in schedule.iterrows():
        f = build_match_features(r, rank_dict, rank_points, stats)
        feats.append(f)
    X = pd.DataFrame(feats, columns=['rank_diff','points_diff','winrate_diff','goal_diff','abs_rank_diff','abs_points_diff'])
    X = X.fillna(X.mean())
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    schedule = schedule.copy()
    schedule['pred'] = preds

    groups = infer_groups(schedule)
    advancers = []
    for g in groups:
        table = {t: {'pts':0,'gd':0,'gf':0,'ga':0} for t in g}
        matches = schedule[schedule['home_team'].isin(g) & schedule['away_team'].isin(g)]
        for _, m in matches.iterrows():
            h = m['home_team']; a = m['away_team']
            p = m['pred']
            if p == 2:
                gh, ga = 1, 0
            elif p == 0:
                gh, ga = 0, 1
            else:
                gh, ga = 1, 1
            table[h]['gf'] += gh; table[h]['ga'] += ga; table[h]['gd'] = table[h]['gf'] - table[h]['ga']
            table[a]['gf'] += ga; table[a]['ga'] += gh; table[a]['gd'] = table[a]['gf'] - table[a]['ga']
            if gh > ga:
                table[h]['pts'] += 3
            elif ga > gh:
                table[a]['pts'] += 3
            else:
                table[h]['pts'] += 1; table[a]['pts'] += 1
        sorted_table = sorted(table.items(), key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gf']), reverse=True)
        if len(sorted_table) >= 2:
            advancers.extend([sorted_table[0][0], sorted_table[1][0]])
        elif len(sorted_table) == 1:
            advancers.append(sorted_table[0][0])

    # knockout
    rng.shuffle(advancers)
    round_teams = advancers
    while len(round_teams) > 1:
        next_round = []
        for i in range(0, len(round_teams), 2):
            t1 = round_teams[i]
            t2 = round_teams[i+1] if i+1 < len(round_teams) else None
            if t2 is None:
                next_round.append(t1); continue
            row = {'home_team': t1, 'away_team': t2}
            f = build_match_features(row, rank_dict, rank_points, stats)
            Xr = scaler.transform([f])
            p = model.predict(Xr)[0]
            if p == 2:
                winner = t1
            elif p == 0:
                winner = t2
            else:
                winner = t1 if rng.random() < 0.5 else t2
            next_round.append(winner)
        round_teams = next_round
    return round_teams[0] if round_teams else None


if __name__ == '__main__':
    import sys
    N = 1000
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except Exception:
            pass
    models_dir = Path('models')
    model = joblib.load(models_dir / 'diff_model.joblib')
    # avoid heavy joblib parallelism inside many small predict calls
    try:
        model.set_params(n_jobs=1)
    except Exception:
        pass
    # debug: print n_jobs
    try:
        print('model n_jobs:', model.get_params().get('n_jobs'))
    except Exception:
        pass
    scaler = joblib.load(models_dir / 'diff_scaler.joblib')

    schedule = pd.read_csv('data/raw/schedule_2026.csv')
    rank = pd.read_csv('data/raw/fifa_ranking_2026-06-08.csv')
    rank_dict = dict(zip(rank['team'], rank['rank']))
    rank_points = dict(zip(rank['team'], rank['points']))
    stats = team_stats_from_matches('data/raw/matches_1930_2022.csv')

    counts = defaultdict(int)
    rng = np.random.default_rng()
    for i in range(N):
        champ = simulate_once(model, scaler, schedule, rank_dict, rank_points, stats, rng)
        counts[champ] += 1
        if (i+1) % 100 == 0:
            print(f'Completed {i+1}/{N} sims')

    total = sum(counts.values())
    probs = [(team, cnt/total) for team, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
    print('\nTop champions:')
    for t, p in probs[:10]:
        print(f'  {t}: {p:.3%}')

    out_dir = Path('models')
    out_dir.mkdir(exist_ok=True)
    df_out = pd.DataFrame([(t,p) for t,p in probs], columns=['team','prob'])
    df_out.to_csv(out_dir / 'champion_probs.csv', index=False)
    print(f"Saved probabilities to {out_dir / 'champion_probs.csv'}")
