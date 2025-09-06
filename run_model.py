#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


def load_squad(path, match_no):
    sheet = f"Match_{match_no}"
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = df.columns.str.strip()
    df = df[df['IsPlaying'].str.upper() == 'PLAYING']
    for col in ['Player Name', 'player', 'Player']:
        if col in df.columns:
            df.rename(columns={col: 'player'}, inplace=True)
            break
    return df


def load_stats(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    if 'team' in df.columns:
        df.rename(columns={'team': 'Team'}, inplace=True)
    if 'player' not in df.columns and 'Player Name' in df.columns:
        df.rename(columns={'Player Name': 'player'}, inplace=True)
    return df


def prepare_features(df):
    """Extract features from the dataset for model training"""
    df['avg'] = pd.to_numeric(df['avg'], errors='coerce').fillna(0)
    df['sr'] = pd.to_numeric(df['sr'], errors='coerce').fillna(0)
    
    numeric_features = ['avg', 'sr', 'matches']
    categorical_features = ['Team', 'Player Type']
    
    df['experience_level'] = np.log1p(df['matches'].fillna(0))
    df['performance_index'] = df['avg'] * df['sr'] / 100

    additional_features = ['experience_level', 'performance_index']
    
    X_num = df[numeric_features + additional_features]
    X_cat = df[categorical_features].fillna('Unknown')
    
    X = pd.concat([X_num, X_cat], axis=1)
    y = df['points_last_3'].fillna(df['avg'])
    return X, y


def train_model(X, y):
    """Train RandomForestRegressor with hyperparameter optimization"""
    numeric_feats = ['avg', 'sr', 'matches', 'experience_level', 'performance_index']
    cat_feats = ['Team', 'Player Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    # Define search space
    param_distributions = {
        'regressor__n_estimators': randint(100, 500),
        'regressor__max_depth': randint(5, 30),
        'regressor__min_samples_split': randint(2, 10),
        'regressor__min_samples_leaf': randint(1, 5),
        'regressor__max_features': ['sqrt', 'log2', uniform(0.3, 0.4)],
        'regressor__bootstrap': [True, False]
    }
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_


def prepare_and_predict(playing_df, stats_df):
    merged = pd.merge(playing_df, stats_df, on=['player', 'Team'], how='left', validate='many_to_many')
    for col in ['matches', 'avg', 'sr', 'points_last_3', 'consistency']:
        merged[col] = merged[col].fillna(0)
    
    # Ensure Player Type exists
    if 'Player Type' not in merged.columns:
        merged['Player Type'] = 'Unknown'
    
    X, y = prepare_features(merged)
    model = train_model(X, y)
    merged['score'] = model.predict(X)
    merged['score'] += merged['consistency'] * 0.1
    merged['score'] += (merged['points_last_3'] - merged['avg']) * 0.2
    return merged


def ensure_player_types(df, required_types):
    """Ensure at least one player of each required type is included"""
    final_indices = []
    remaining_types = required_types.copy()
    
    # First pass: try to get one of each required type from the bottom ranked players
    for player_type in required_types:
        type_players = df[df['Player Type'] == player_type]
        if not type_players.empty:
            # Get the worst player of this type (lowest score)
            idx = type_players.index[0]
            final_indices.append(idx)
            remaining_types.remove(player_type)
    
    # If we still have unfilled roles, find players regardless of rank
    if remaining_types:
        remaining_df = df.drop(final_indices)
        for player_type in remaining_types:
            type_players = remaining_df[remaining_df['Player Type'] == player_type]
            if not type_players.empty:
                idx = type_players.index[0]
                final_indices.append(idx)
    
    # Fill the rest of the 11 with the original ranking logic
    remaining_spots = 11 - len(final_indices)
    if remaining_spots > 0:
        remaining_df = df.drop(final_indices)
        for idx in remaining_df.head(remaining_spots).index:
            final_indices.append(idx)
    
    # Create final DataFrame and sort by score
    final_df = df.loc[final_indices].sort_values('score', ascending=True).reset_index(drop=True)
    return final_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('match_number', type=int, help="Match number, e.g. 30")
    p.add_argument('--squad-file', default='/app/data/SquadPlayerNames_IndianT20League.xlsx')
    p.add_argument('--stats-file', default='/app/data/ipl_2025_gameathon.xlsx')
    args = p.parse_args()

    playing = load_squad(args.squad_file, args.match_number)
    stats = load_stats(args.stats_file)

    for col in ['matches', 'avg', 'sr', 'points_last_3', 'consistency']:
        if col not in stats.columns:
            stats[col] = 0

    result = prepare_and_predict(playing, stats)
    result['score'] = pd.to_numeric(result['score'], errors='coerce')

    result = result.sort_values('score', ascending=False)
    result = result.drop_duplicates(subset=['player'], keep='first')
    result = result.sort_values('score', ascending=True)
    
    # Get playing 11 with required player types
    required_types = ['WK', 'BAT', 'BOWL', 'ALL']
    top_11 = ensure_player_types(result, required_types.copy())
    
    # Get remaining 4 players for backup
    used_players = set(top_11['player'])
    backup_players = result[~result['player'].isin(used_players)].head(4).reset_index(drop=True)
    
    # Create final dataframe with blank row separator
    out = top_11.copy()
    out['C/VC'] = 'NA'  # Initialize all with NA
    
    # Assign captain and vice-captain
    out.loc[0, 'C/VC'] = 'C'
    captain_team = out.loc[0, 'Team']

    vc_idx = None
    for idx in range(1, len(out)):
        if out.loc[idx, 'Team'] != captain_team:
            vc_idx = idx
            break
    if vc_idx is None and len(out) > 1:
        vc_idx = 1
    if vc_idx is not None and vc_idx != 1:
        vc_row = out.iloc[[vc_idx]]
        remaining = out.drop(vc_idx)
        out = pd.concat([out.iloc[[0]], vc_row, remaining.iloc[1:]]).reset_index(drop=True)
    if len(out) > 1:
        out.loc[1, 'C/VC'] = 'VC'

    # Create blank row and append backup players
    blank_row = pd.DataFrame({'player': [''], 'Team': [''], 'C/VC': ['']})
    backup_df = backup_players[['player', 'Team']].copy()
    backup_df['C/VC'] = 'NA'  # Set backup players to NA
    
    out = pd.concat([
        out[['player', 'Team', 'C/VC']],
        blank_row,
        backup_df
    ]).reset_index(drop=True)
    
    # Final formatting
    out = out.rename(columns={'player': 'Player name'})
    out_path = '/Downloads/The_Gladiators_output.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"✅ The_Gladiators_output.csv saved at {out_path}")

import warnings
warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

if __name__ == '__main__':
    main()