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
    # build graph from group-stage matches and find connected components
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


def build_match_features(row, rank_dict, stats):
    home = row['home_team']
    away = row['away_team']
    home_rank = rank_dict.get(home, 100)
    away_rank = rank_dict.get(away, 100)
    home_points = rank_dict_points.get(home, 0)
    away_points = rank_dict_points.get(away, 0)
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


if __name__ == '__main__':
    models_dir = Path('models')
    model = joblib.load(models_dir / 'diff_model.joblib')
    scaler = joblib.load(models_dir / 'diff_scaler.joblib')

    schedule = pd.read_csv('data/raw/schedule_2026.csv')
    rank = pd.read_csv('data/raw/fifa_ranking_2026-06-08.csv')
    rank_dict = dict(zip(rank['team'], rank['rank']))
    rank_dict_points = dict(zip(rank['team'], rank['points']))

    stats = team_stats_from_matches('data/raw/matches_1930_2022.csv')

    # predict each scheduled match
    feats = []
    for _, r in schedule.iterrows():
        f = build_match_features(r, rank_dict, stats)
        feats.append(f)

    X = pd.DataFrame(feats, columns=['rank_diff','points_diff','winrate_diff','goal_diff','abs_rank_diff','abs_points_diff'])
    X = X.fillna(X.mean())
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)
    preds = model.predict(Xs)

    schedule['pred'] = preds
    schedule['prob_away'] = probs[:,0]
    schedule['prob_draw'] = probs[:,1]
    schedule['prob_home'] = probs[:,2]

    # infer groups and compute standings
    groups = infer_groups(schedule)
    print(f'Inferred {len(groups)} groups')

    group_standings = {}
    for g in groups:
        # initialize
        table = {t: {'pts':0,'gd':0,'gf':0,'ga':0} for t in g}
        matches = schedule[schedule['home_team'].isin(g) & schedule['away_team'].isin(g)]
        for _, m in matches.iterrows():
            h = m['home_team']; a = m['away_team']
            p = m['pred']
            # assign simple scores
            if p == 2:  # home win
                gh, ga = 1, 0
            elif p == 0:  # away win
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
        # sort table
        sorted_table = sorted(table.items(), key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gf']), reverse=True)
        group_standings[','.join(g)] = sorted_table

    # pick top 2 from each group to advance
    advancers = []
    for g, table in group_standings.items():
        top2 = [table[0][0], table[1][0]]
        advancers.extend(top2)
    print(f'Advanced teams: {len(advancers)}')

    # simple single-elimination bracket: random pair and simulate until one left
    np.random.seed(42)
    np.random.shuffle(advancers)
    round_teams = advancers
    round_num = 1
    while len(round_teams) > 1:
        print(f'Round {round_num}: {len(round_teams)} teams')
        next_round = []
        for i in range(0, len(round_teams), 2):
            t1 = round_teams[i]
            t2 = round_teams[i+1] if i+1 < len(round_teams) else None
            if t2 is None:
                next_round.append(t1); continue
            # build a fake row for t1 home vs t2 away
            row = {'home_team': t1, 'away_team': t2}
            f = build_match_features(row, rank_dict, stats)
            Xr = scaler.transform([f])
            p = model.predict(Xr)[0]
            winner = t1 if p == 2 else (t2 if p == 0 else (t1 if np.random.rand() < 0.5 else t2))
            next_round.append(winner)
        round_teams = next_round
        round_num += 1

    champion = round_teams[0] if round_teams else None
    print('\nPredicted champion:', champion)
