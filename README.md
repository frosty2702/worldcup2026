"# worldcup2026

This project predicts World Cup match outcomes and ranks teams. To improve ranking predictions, collect richer and more up-to-date data and run the pipeline below.

Data to collect (recommended):
- Historical match results (include qualifiers & friendlies) — place CSVs in data/raw/
- FIFA rankings and Elo ratings (e.g., eloratings.net)
- Team-level stats: points, confederation, recent form (last 5–10 matches)
- Player-level data: minutes, goals, injuries, market value (optional)
- Fixtures, venue, rest days, travel distance

Pipeline (run in repository root):
1. python scripts/clean_data.py   # creates data/cleaned_matches.csv
2. python scripts/team_stats.py   # creates data/team_stats.csv
3. python scripts/merge_team_stats.py  # creates data/final_dataset_v2.csv
4. python scripts/train_ranked_model.py  # trains model and prints accuracy

How to add a new data source:
- Put raw CSV/API dump in data/raw/
- Add an ingestion script under scripts/ (e.g., scripts/ingest_source.py) that normalizes column names
- Update merge or preprocessing scripts to include new features

Suggested additional features for better rankings:
- Head-to-head metrics, recent-form weighting, Elo ratings
- Player availability/injury flags and aggregated player impact
- Home/away adjustments, venue/weather, rest days, travel distance

Contributions welcome — open an issue or PR with new data sources or ingestion scripts.
" 
