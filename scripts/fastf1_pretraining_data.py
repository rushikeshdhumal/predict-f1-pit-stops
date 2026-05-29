"""
FastF1 pretraining data pipeline for NB26.

Downloads real F1 lap data (2022-2025 seasons) using the FastF1 library,
engineers the same feature set used in competition training, and saves the
processed dataset for local pretraining of the Hybrid GRU+FC model.

Usage:
    cd c:/Repos/predict-f1-pit-stops
    .venv\\Scripts\\Activate.ps1
    pip install fastf1
    python scripts/fastf1_pretraining_data.py

Outputs (saved to data/fastf1/):
    - f1_laps_raw.parquet        — raw lap data from all 2022-2025 races
    - f1_pretrain_features.parquet — engineered features (same schema as competition)
    - pretrain_data_summary.txt  — stats per season/race

Runtime: ~2-4 hours (FastF1 downloads + caches session data; ~66-72 race sessions).
The cache lives at data/fastf1/cache/ and persists between runs.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Project root detection ────────────────────────────────────────────────────
cwd = Path(__file__).resolve()
while cwd.name != 'predict-f1-pit-stops' and cwd.parent != cwd:
    cwd = cwd.parent
PROJECT_ROOT = cwd
DATA_DIR     = PROJECT_ROOT / 'data' / 'fastf1'
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_DIR / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

print(f'Project root : {PROJECT_ROOT}')
print(f'FastF1 cache : {CACHE_DIR}')

# ── FastF1 import ─────────────────────────────────────────────────────────────
try:
    import fastf1
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    print(f'FastF1 {fastf1.__version__} ready, cache enabled.')
except ImportError:
    print('ERROR: fastf1 not installed. Run: pip install fastf1')
    sys.exit(1)

# ── Seasons and rounds to download ───────────────────────────────────────────
SEASONS = [2022, 2023, 2024, 2025]
# Race sessions only (not qualifying, practice, sprint)
SESSION_TYPE = 'R'

# Tyre cliff thresholds from competition CLAUDE.md (for compound normalization)
CLIFF_THRESHOLDS = {'SOFT': 13, 'MEDIUM': 49, 'HARD': 61,
                    'INTERMEDIATE': 13, 'WET': 5}
COMPOUND_ORDINAL  = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 0, 'WET': 0}


def extract_lap_features(session):
    """Extract and engineer features from a FastF1 session."""
    laps = session.laps.copy()

    # ── Basic filtering ───────────────────────────────────────────────────────
    # Drop laps with missing essential fields
    essential = ['Driver', 'Compound', 'TyreLife', 'LapTime', 'LapNumber',
                 'Stint', 'Position']
    laps = laps.dropna(subset=essential).copy()
    if len(laps) == 0:
        return None

    # ── Core columns ──────────────────────────────────────────────────────────
    laps['LapTime (s)']  = laps['LapTime'].dt.total_seconds()
    laps['Race']         = session.event['EventName']
    laps['Year']         = int(session.event['EventDate'].year)
    laps['LapNumber']    = laps['LapNumber'].astype(int)
    laps['Stint']        = laps['Stint'].astype(int)
    laps['TyreLife']     = laps['TyreLife'].fillna(0).astype(float)
    laps['Position']     = laps['Position'].fillna(laps['Position'].median()).astype(float)

    # Standardize compound names to match competition dataset
    compound_map = {
        'SOFT': 'SOFT', 'MEDIUM': 'MEDIUM', 'HARD': 'HARD',
        'INTERMEDIATE': 'INTERMEDIATE', 'WET': 'WET',
        'SUPERSOFT': 'SOFT', 'ULTRASOFT': 'SOFT', 'HYPERSOFT': 'SOFT',
        'C1': 'HARD', 'C2': 'HARD', 'C3': 'MEDIUM',
        'C4': 'SOFT', 'C5': 'SOFT', 'UNKNOWN': None,
    }
    laps['Compound'] = laps['Compound'].str.upper().map(
        lambda x: compound_map.get(x, x))
    laps = laps[laps['Compound'].notna()].copy()

    # ── PitStop flag ──────────────────────────────────────────────────────────
    # PitInTime is non-null when the driver entered the pits at the end of this lap
    laps['PitStop'] = laps['PitInTime'].notna().astype(int)

    # ── Sort by driver + lap ──────────────────────────────────────────────────
    laps = laps.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)

    # ── PitNextLap (target) ───────────────────────────────────────────────────
    laps['PitNextLap'] = (
        laps.groupby('Driver')['PitStop'].shift(-1).fillna(0).astype(int)
    )

    # ── RaceProgress, laps_remaining ─────────────────────────────────────────
    max_lap = laps.groupby('Driver')['LapNumber'].transform('max')
    total_race_laps = laps['LapNumber'].max()
    laps['RaceProgress']   = laps['LapNumber'] / total_race_laps
    laps['laps_remaining'] = 1.0 - laps['RaceProgress']

    # ── Position_Change ───────────────────────────────────────────────────────
    laps['Position_Change'] = (
        laps.groupby('Driver')['Position'].diff().fillna(0)
    )

    # ── LapTime_Delta (vs driver reference pace) ──────────────────────────────
    # Use driver's median of first 5 laps in each stint as reference pace
    def driver_reference_pace(grp):
        ref = grp.groupby('Stint')['LapTime (s)'].transform(
            lambda x: x.head(5).median())
        return ref

    laps['driver_ref_pace'] = laps.groupby('Driver', group_keys=False).apply(
        driver_reference_pace)
    laps['LapTime_Delta'] = laps['LapTime (s)'] - laps['driver_ref_pace']
    laps['LapTime_Delta'] = laps['LapTime_Delta'].fillna(0).clip(-20, 30)

    # ── Cumulative_Degradation (within stint, winsorized later) ──────────────
    laps['Cumulative_Degradation'] = (
        laps.groupby(['Driver', 'Stint'])['LapTime_Delta']
        .transform(pd.Series.cumsum)
    )

    # Winsorize at competition percentiles: [-205, +122]
    laps['Cumulative_Degradation_winsorized'] = (
        laps['Cumulative_Degradation'].clip(-205, 122)
    )

    # ── Tyre compound features ────────────────────────────────────────────────
    laps['is_wet_tyre']       = laps['Compound'].isin(['INTERMEDIATE', 'WET']).astype(int)
    laps['compound_ordinal']  = laps['Compound'].map(COMPOUND_ORDINAL).fillna(0)
    laps['TyreLife_normalized_by_compound'] = (
        laps['TyreLife'] / laps['Compound'].map(CLIFF_THRESHOLDS).fillna(13)
    )
    laps['TyreLife_sq'] = laps['TyreLife'] ** 2

    # ── Degradation rate ─────────────────────────────────────────────────────
    laps['Degradation_rate'] = (
        laps['Cumulative_Degradation_winsorized'] /
        laps['TyreLife'].replace(0, np.nan)
    ).fillna(0)

    # ── Lag features (within Race+Driver+Stint) ───────────────────────────────
    stint_group = laps.groupby(['Driver', 'Stint'])
    for lag in [1, 2, 3]:
        laps[f'LapTime_lag{lag}']       = stint_group['LapTime (s)'].shift(lag).fillna(0)
        laps[f'LapTime_Delta_lag{lag}'] = stint_group['LapTime_Delta'].shift(lag).fillna(0)

    laps['Degradation_acceleration'] = (
        stint_group['Cumulative_Degradation_winsorized'].diff().fillna(0)
    )

    # ── Rolling features ──────────────────────────────────────────────────────
    for w in [3, 5]:
        laps[f'LapTime_rolling_mean_{w}'] = (
            stint_group['LapTime (s)'].transform(
                lambda x: x.rolling(w, min_periods=1).mean()))
        laps[f'LapTime_rolling_std_{w}'] = (
            stint_group['LapTime (s)'].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)))
        laps[f'Degradation_rolling_slope_{w}'] = (
            stint_group['Cumulative_Degradation_winsorized'].transform(
                lambda x: x.rolling(w, min_periods=1).mean()))

    # ── Position volatility ───────────────────────────────────────────────────
    laps['abs_position_change'] = laps['Position_Change'].abs()
    laps['pos_change_rolling_std_3'] = (
        laps.groupby('Driver')['Position_Change']
        .transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    )

    # ── PitStop lag ───────────────────────────────────────────────────────────
    laps['PitStop_lag1'] = (
        laps.groupby('Driver')['PitStop'].shift(1).fillna(0)
    )

    # ── Pit window ───────────────────────────────────────────────────────────
    laps['prime_pit_window']    = (
        ((laps['RaceProgress'] >= 0.4) & (laps['RaceProgress'] < 0.7)).astype(int))
    laps['prime_window_x_compound'] = (
        laps['prime_pit_window'] * laps['compound_ordinal'])

    # ── Interaction features ──────────────────────────────────────────────────
    laps['TyreLife_x_laps_remaining']     = laps['TyreLife'] * laps['laps_remaining']
    laps['Degradation_x_RaceProgress']   = (
        laps['Cumulative_Degradation_winsorized'] * laps['RaceProgress'])
    laps['Position_x_RaceProgress']      = laps['Position'] * laps['RaceProgress']
    laps['TyreLife_x_compound_ordinal']  = laps['TyreLife'] * laps['compound_ordinal']
    laps['Stint_x_compound_ordinal']     = laps['Stint'] * laps['compound_ordinal']
    laps['TyreLife_x_cmpd_x_laps_rem']  = (
        laps['TyreLife'] * laps['compound_ordinal'] * laps['laps_remaining'])

    # ── Laps to driver end ────────────────────────────────────────────────────
    laps['laps_to_driver_end'] = max_lap - laps['LapNumber']

    # ── Field-level features ─────────────────────────────────────────────────
    field_med = (
        laps.groupby('LapNumber')['LapTime (s)']
        .median().rename('field_median_laptime')
    )
    laps = laps.join(field_med, on='LapNumber')
    laps['laptime_vs_field'] = laps['LapTime (s)'] - laps['field_median_laptime']
    laps['field_pace_change'] = (
        laps.groupby('Driver')['field_median_laptime'].diff().fillna(0)
    )

    # ── Testing session flag (never True for real races) ─────────────────────
    laps['is_testing_session'] = 0

    # ── Target encoding: use race-level mean for pretraining (no fold split) ──
    pit_rate_overall = laps['PitNextLap'].mean()
    race_pit_rate    = laps.groupby('Race')['PitNextLap'].mean()
    driver_pit_rate  = laps.groupby('Driver')['PitNextLap'].mean()

    laps['Race_target_encoded']   = laps['Race'].map(race_pit_rate).fillna(pit_rate_overall)
    laps['Driver_target_encoded'] = laps['Driver'].map(driver_pit_rate).fillna(pit_rate_overall)

    driver_avg_stint = (
        laps[laps['PitStop'] == 1]
        .groupby('Driver')['TyreLife'].mean()
    )
    laps['Driver_avg_stint_length'] = (
        laps['Driver'].map(driver_avg_stint).fillna(laps['TyreLife'].mean()))

    # Race_Year target encoding
    ry_key = laps['Race'] + '_' + laps['Year'].astype(str)
    ry_pit_rate = laps.groupby(ry_key)['PitNextLap'].mean()
    laps['Race_Year_target_encoded'] = ry_key.map(ry_pit_rate).fillna(pit_rate_overall)

    # Driver_Compound target encoding
    dc_key = laps['Driver'] + '_' + laps['Compound']
    dc_pit_rate = laps.groupby(dc_key)['PitNextLap'].mean()
    laps['Driver_Compound_target_encoded'] = dc_key.map(dc_pit_rate).fillna(pit_rate_overall)

    # TyreLife vs driver typical (when they pit for this compound)
    dct_typical = (
        laps[laps['PitStop'] == 1]
        .groupby(['Driver', 'Compound'])['TyreLife'].median()
    )
    def tyrelife_vs_typical(row):
        try:
            return row['TyreLife'] - dct_typical.loc[(row['Driver'], row['Compound'])]
        except KeyError:
            return 0.0

    laps['TyreLife_vs_driver_typical'] = laps.apply(tyrelife_vs_typical, axis=1)

    return laps


# ── Main download loop ────────────────────────────────────────────────────────
def main():
    all_laps = []
    summary_lines = []

    for year in SEASONS:
        print(f'\n{"="*60}\nSeason {year}\n{"="*60}')
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        race_rounds = schedule[schedule['EventFormat'] != 'testing']['RoundNumber'].tolist()

        for round_num in race_rounds:
            event_name = schedule.loc[
                schedule['RoundNumber'] == round_num, 'EventName'].values[0]
            print(f'  [{year} R{round_num:02d}] {event_name} ... ', end='', flush=True)

            try:
                t0      = time.time()
                session = fastf1.get_session(year, round_num, SESSION_TYPE)
                session.load(telemetry=False, weather=False, messages=False)
                laps_df = extract_lap_features(session)
                elapsed = time.time() - t0

                if laps_df is None or len(laps_df) < 50:
                    print(f'SKIP (too few laps: {len(laps_df) if laps_df is not None else 0})')
                    continue

                all_laps.append(laps_df)
                n_pos = laps_df['PitNextLap'].sum()
                rate  = laps_df['PitNextLap'].mean()
                summary_lines.append(
                    f'{year} R{round_num:02d} {event_name}: {len(laps_df)} laps, '
                    f'{n_pos} pits ({rate:.1%}) [{elapsed:.0f}s]')
                print(f'OK — {len(laps_df)} laps, pit rate {rate:.1%} ({elapsed:.0f}s)')

            except Exception as e:
                print(f'ERROR — {e}')
                summary_lines.append(f'{year} R{round_num:02d} {event_name}: ERROR — {e}')

    if not all_laps:
        print('ERROR: No lap data collected.')
        return

    combined = pd.concat(all_laps, ignore_index=True)
    print(f'\nTotal rows collected: {len(combined):,}')
    print(f'Pit rate overall: {combined["PitNextLap"].mean():.2%}')
    print(f'Races collected: {combined.groupby(["Race","Year"]).ngroups}')

    # ── Save raw ──────────────────────────────────────────────────────────────
    raw_path = DATA_DIR / 'f1_laps_raw.parquet'
    combined.to_parquet(raw_path, index=False)
    print(f'\nSaved raw: {raw_path}  ({len(combined):,} rows)')

    # ── Select final feature columns matching competition schema ──────────────
    FEAT_COLS = [
        'TyreLife_normalized_by_compound', 'TyreLife_sq', 'is_wet_tyre', 'compound_ordinal',
        'laps_remaining', 'is_testing_session', 'Stint', 'Position',
        'Degradation_rate', 'Degradation_acceleration', 'Cumulative_Degradation_winsorized',
        'LapTime_lag1', 'LapTime_lag2', 'LapTime_lag3',
        'LapTime_Delta_lag1', 'LapTime_Delta_lag2', 'LapTime_Delta_lag3',
        'LapTime_rolling_mean_3', 'LapTime_rolling_mean_5',
        'LapTime_rolling_std_3', 'LapTime_rolling_std_5',
        'Degradation_rolling_slope_3', 'Degradation_rolling_slope_5',
        'TyreLife_x_laps_remaining', 'Degradation_x_RaceProgress', 'Position_x_RaceProgress',
        'TyreLife_x_compound_ordinal', 'Stint_x_compound_ordinal', 'TyreLife_x_cmpd_x_laps_rem',
        'prime_pit_window', 'prime_window_x_compound',
        'abs_position_change', 'pos_change_rolling_std_3', 'PitStop_lag1',
        'laps_to_driver_end', 'field_median_laptime', 'laptime_vs_field', 'field_pace_change',
        # Additional competition features (target encodings)
        'Race_target_encoded', 'Driver_target_encoded', 'Driver_avg_stint_length',
        'Race_Year_target_encoded', 'Driver_Compound_target_encoded',
        'TyreLife_vs_driver_typical',
    ]
    SEQ_COLS = ['LapTime (s)', 'TyreLife', 'Cumulative_Degradation_winsorized',
                'LapTime_Delta', 'Position', 'PitStop']
    META_COLS = ['Driver', 'Race', 'Year', 'Compound', 'LapNumber', 'PitStop', 'PitNextLap',
                 'RaceProgress', 'LapTime_Delta']

    # Check missing columns
    all_needed = FEAT_COLS + SEQ_COLS + META_COLS
    missing = [c for c in all_needed if c not in combined.columns]
    if missing:
        print(f'WARNING: Missing columns: {missing}')

    available = [c for c in list(dict.fromkeys(all_needed)) if c in combined.columns]
    features_df = combined[available].copy()
    features_df = features_df.fillna(0)  # fill any remaining NaNs

    feat_path = DATA_DIR / 'f1_pretrain_features.parquet'
    features_df.to_parquet(feat_path, index=False)
    print(f'Saved features: {feat_path}  ({len(features_df):,} rows, {features_df.shape[1]} cols)')

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = DATA_DIR / 'pretrain_data_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f'FastF1 pretraining data — {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n')
        f.write(f'Seasons: {SEASONS}\n')
        f.write(f'Total rows: {len(combined):,}\n')
        f.write(f'Total races: {combined.groupby(["Race","Year"]).ngroups}\n')
        f.write(f'Pit rate: {combined["PitNextLap"].mean():.2%}\n')
        f.write(f'Feature columns: {len(available)}\n\n')
        f.write('Per-race summary:\n')
        for line in summary_lines:
            f.write(line + '\n')

    print(f'\nSummary: {summary_path}')
    print('\nNext step: run scripts/pretrain_hybrid.py to pretrain Hybrid GRU+FC on this data,')
    print('then upload the pretrained weights to Kaggle as a dataset for NB26.')


if __name__ == '__main__':
    main()
