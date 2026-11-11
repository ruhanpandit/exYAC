"""
YAC Available: Measuring open-field opportunity at the catch point.
NFL Big Data Bowl Regional Hackathon — November 2026
"""

import os
import sys
import re
import time
import argparse
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ── Constants (tunable) ─────────────────────────────────────────────
RECEIVER_SPEED = 8.0        # yd/s assumed for YAC-available race
DEFENDER_SPEED = 7.5        # yd/s assumed for defenders (post-catch closing)
DEFENDER_BREAK_SPEED = 7.5  # yd/s defenders break toward catch point during flight
DEFENDER_REACTION = 0.3     # seconds before defenders react to throw
TACKLE_RADIUS = 1.0         # yards — defender closes lane if within this distance
RECEIVER_CATCH_DELAY = 0.4  # seconds — secure catch + turn before running
BALL_SPEED = 18.0           # yd/s assumed pass speed
MIN_FLIGHT_TIME = 0.3       # seconds
MAX_FLIGHT_TIME = 3.0       # seconds
LANE_ANGLES = np.linspace(-20, 20, 5)  # degrees off downfield (narrowed)
LANE_STEP = 0.5             # yards per step along lane
MAX_LANE_YARDS = 50         # max yards to check along any lane
ENDZONE_X = 110.0           # right end zone starts here
MIN_RECEPTIONS = 15         # min plays for leaderboard
TOP_N_LANES = 2             # average top N lanes instead of pure max

# ── Paths ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data')
args, _ = parser.parse_known_args()
DATA_DIR = args.data_dir
TRACKING_DIR = os.path.join(DATA_DIR, 'tracking')
CACHE_DIR = 'data_cache'
OUTPUT_DIR = 'outputs'
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper: download nflverse PBP ───────────────────────────────────
def get_nflverse_pbp():
    cache_path = os.path.join(CACHE_DIR, 'play_by_play_2021.csv.gz')
    if not os.path.exists(cache_path):
        url = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.csv.gz'
        print(f'  Downloading nflverse PBP → {cache_path} …')
        urllib.request.urlretrieve(url, cache_path)
    cols = ['old_game_id', 'play_id', 'air_yards', 'yards_after_catch',
            'xyac_mean_yardage', 'complete_pass']
    pbp = pd.read_csv(cache_path, usecols=cols, low_memory=False)
    pbp = pbp.rename(columns={'old_game_id': 'gameId', 'play_id': 'playId'})
    pbp['gameId'] = pbp['gameId'].astype('Int64')
    pbp['playId'] = pbp['playId'].astype('Int64')
    return pbp

# ── Helper: parse target receiver from playDescription ──────────────
TARGET_RE = re.compile(
    r'(?:pass\s+\S+\s+\S+\s+to|intended\s+for)\s+'
    r"([A-Z][a-zA-Z'\.\-]*\.[A-Za-z' \.\-]+?)(?=\s+to\s|\s+for\s|\.|\,|\s+\(|\s+ran|\s+pushed|\s+\[|$)"
)

def parse_target(desc):
    m = TARGET_RE.search(desc)
    if m:
        return m.group(1).strip()
    return None

def make_short_name(display_name):
    """Convert 'Chris Godwin' → 'C.Godwin'"""
    parts = display_name.strip().split()
    if len(parts) < 2:
        return display_name
    first_initial = parts[0][0]
    last = parts[-1]
    return f"{first_initial}.{last}"

def make_last_name(display_name):
    return display_name.strip().split()[-1]

# ── Helper: normalize coordinates ───────────────────────────────────
def normalize_tracking(df):
    """In-place normalize so offense always moves toward +x."""
    left = df['playDirection'] == 'left'
    df.loc[left, 'x'] = 120.0 - df.loc[left, 'x']
    df.loc[left, 'y'] = 53.3 - df.loc[left, 'y']
    df.loc[left, 'dir'] = (df.loc[left, 'dir'] + 180.0) % 360.0
    # 'o' not loaded (not needed); skip normalizing it
    return df

# ── Helper: compute velocities ──────────────────────────────────────
def add_velocity(df):
    dir_rad = np.deg2rad(df['dir'].values)
    df['vx'] = df['s'].values * np.sin(dir_rad)
    df['vy'] = df['s'].values * np.cos(dir_rad)
    return df

# ── Project defenders during ball flight (break on ball) ────────────
def project_defenders_break(def_x, def_y, catch_x, catch_y, t_f):
    """
    During ball flight, after DEFENDER_REACTION s, each defender moves
    toward the catch point at DEFENDER_BREAK_SPEED. They stop once they
    reach the catch point (don't overshoot).
    Returns: proj_def_x, proj_def_y (arrays)
    """
    move_time = max(t_f - DEFENDER_REACTION, 0.0)
    # Direction from each defender to catch point
    dx = catch_x - def_x
    dy = catch_y - def_y
    dist_to_catch = np.sqrt(dx**2 + dy**2)
    dist_to_catch = np.maximum(dist_to_catch, 1e-6)  # avoid div by zero

    # Max distance they can cover
    max_move = DEFENDER_BREAK_SPEED * move_time
    # Don't overshoot past catch point
    actual_move = np.minimum(max_move, dist_to_catch)

    proj_x = def_x + (dx / dist_to_catch) * actual_move
    proj_y = def_y + (dy / dist_to_catch) * actual_move
    proj_x = np.clip(proj_x, 0, 120)
    proj_y = np.clip(proj_y, 0, 53.3)
    return proj_x, proj_y


# ── YAC Available computation ───────────────────────────────────────
def compute_yac_available(recv_x, recv_y, recv_vx, recv_vy,
                          def_x, def_y, def_vx, def_vy,
                          catch_x, catch_y, los_x):
    """
    recv_*: scalars for the receiver at throw frame
    def_*: arrays (n_defenders,) for defenders at throw frame
    catch_x, catch_y: estimated catch point
    los_x: line of scrimmage x
    Returns: yac_available, best_angle, end_x, end_y
    """
    # Flight time
    ball_dist = max(abs(catch_x - recv_x), 1.0)
    t_f = np.clip(ball_dist / BALL_SPEED, MIN_FLIGHT_TIME, MAX_FLIGHT_TIME)

    # Project defenders: break toward catch point during flight
    proj_def_x, proj_def_y = project_defenders_break(
        def_x, def_y, catch_x, catch_y, t_f)

    # Evaluate each lane angle
    lane_gains = []
    lane_ends = []

    for angle_deg in LANE_ANGLES:
        angle_rad = np.deg2rad(angle_deg)
        dx = np.cos(angle_rad)  # downfield component per unit step
        dy = np.sin(angle_rad)  # lateral component per unit step

        lane_gain = 0.0
        end_x, end_y = catch_x, catch_y

        n_steps = int(MAX_LANE_YARDS / LANE_STEP)
        for step_i in range(1, n_steps + 1):
            dist_along = step_i * LANE_STEP
            px = catch_x + dist_along * dx
            py = catch_y + dist_along * dy

            # Out of bounds or past end zone
            if py < 0 or py > 53.3 or px > ENDZONE_X:
                px = min(px, ENDZONE_X)
                lane_gain = px - catch_x
                end_x, end_y = px, py
                break

            # Receiver arrival time from catch point (includes catch-and-turn delay)
            recv_time = RECEIVER_CATCH_DELAY + dist_along / RECEIVER_SPEED

            # Defender arrival times (with tackle radius)
            d_dists = np.sqrt((proj_def_x - px)**2 + (proj_def_y - py)**2)
            effective_dists = np.maximum(d_dists - TACKLE_RADIUS, 0.0)
            def_times = DEFENDER_REACTION + effective_dists / DEFENDER_SPEED
            min_def_time = np.min(def_times)

            if min_def_time <= recv_time:
                break
            else:
                lane_gain = px - catch_x
                end_x, end_y = px, py

        lane_gains.append(lane_gain)
        lane_ends.append((end_x, end_y))

    # Mean of top N lanes instead of pure max
    lane_gains = np.array(lane_gains)
    sorted_idx = np.argsort(lane_gains)[::-1]
    top_n = min(TOP_N_LANES, len(lane_gains))
    yac_avail = np.mean(lane_gains[sorted_idx[:top_n]])

    best_idx = sorted_idx[0]
    best_angle = LANE_ANGLES[best_idx]
    best_end_x, best_end_y = lane_ends[best_idx]

    return yac_avail, best_angle, best_end_x, best_end_y


# ── MAIN PIPELINE ────────────────────────────────────────────────────
t0 = time.time()
print("═══ YAC Available Pipeline ═══")

# 1. Load static tables
print("\n[1] Loading static tables …")
games = pd.read_csv(os.path.join(DATA_DIR, 'games.csv'))
plays = pd.read_csv(os.path.join(DATA_DIR, 'plays.csv'))
players = pd.read_csv(os.path.join(DATA_DIR, 'players.csv'))
pff = pd.read_csv(os.path.join(DATA_DIR, 'pffScoutingData.csv'))

print(f"  {len(games)} games, {len(plays)} plays, {len(players)} players")

# 2. Load nflverse PBP
print("\n[2] Loading nflverse play-by-play …")
pbp = get_nflverse_pbp()
plays = plays.merge(pbp, on=['gameId', 'playId'], how='left')
n_match = plays['air_yards'].notna().sum()
print(f"  Merged nflverse: {n_match}/{len(plays)} plays have air_yards")

# 3. Filter to completions with YAC
completions = plays[plays['passResult'] == 'C'].copy()
completions = completions[completions['yards_after_catch'].notna()].copy()
completions = completions[completions['air_yards'].notna()].copy()
print(f"  Completions with YAC and air_yards: {len(completions)}")

# 4. Build target receiver lookup
print("\n[3] Parsing target receivers …")
# Route runners per play
route_runners = pff[pff['pff_role'] == 'Pass Route'][['gameId', 'playId', 'nflId']].copy()
route_runners = route_runners.merge(players[['nflId', 'displayName', 'officialPosition']], on='nflId', how='left')
route_runners['shortName'] = route_runners['displayName'].apply(make_short_name)
route_runners['lastName'] = route_runners['displayName'].apply(make_last_name)

completions['parsedTarget'] = completions['playDescription'].apply(parse_target)
print(f"  Parsed a target name from {completions['parsedTarget'].notna().sum()}/{len(completions)} completions")

# Match parsed names to route runners
target_map = {}  # (gameId, playId) -> nflId
matched = 0
unmatched_names = []

for _, row in completions.iterrows():
    gid, pid = row['gameId'], row['playId']
    parsed = row['parsedTarget']
    if parsed is None:
        continue

    rr = route_runners[(route_runners['gameId'] == gid) & (route_runners['playId'] == pid)]
    if len(rr) == 0:
        continue

    # Try exact short name match
    exact = rr[rr['shortName'] == parsed]
    if len(exact) == 1:
        target_map[(gid, pid)] = exact.iloc[0]['nflId']
        matched += 1
        continue

    # Try last name match
    parsed_last = parsed.split('.')[-1].strip()
    last_match = rr[rr['lastName'] == parsed_last]
    if len(last_match) == 1:
        target_map[(gid, pid)] = last_match.iloc[0]['nflId']
        matched += 1
        continue

    # Try contains on last name
    contains = rr[rr['lastName'].str.contains(parsed_last, case=False, na=False)]
    if len(contains) == 1:
        target_map[(gid, pid)] = contains.iloc[0]['nflId']
        matched += 1
        continue

    unmatched_names.append(parsed)

print(f"  Matched target receiver: {matched}/{completions['parsedTarget'].notna().sum()} "
      f"({100*matched/max(1,completions['parsedTarget'].notna().sum()):.1f}%)")
if unmatched_names:
    from collections import Counter
    top_unmatched = Counter(unmatched_names).most_common(5)
    print(f"  Top unmatched: {top_unmatched}")

# Filter completions to those with a matched target
completions['targetNflId'] = completions.apply(
    lambda r: target_map.get((r['gameId'], r['playId']), np.nan), axis=1)
completions = completions[completions['targetNflId'].notna()].copy()
completions['targetNflId'] = completions['targetNflId'].astype(int)
print(f"  Completions with matched target: {len(completions)}")

# 5. Process tracking game by game
print("\n[4] Processing tracking data …")
game_ids = sorted(completions['gameId'].unique())
print(f"  Games to process: {len(game_ids)}")

results = []
tracking_cols = ['gameId', 'playId', 'nflId', 'frameId', 'team',
                 'playDirection', 'x', 'y', 's', 'dir', 'event']

for gi, gid in enumerate(game_ids):
    fname = os.path.join(TRACKING_DIR, f'tracking_{gid}.csv')
    if not os.path.exists(fname):
        continue

    trk = pd.read_csv(fname, usecols=tracking_cols)
    trk = normalize_tracking(trk)
    trk = add_velocity(trk)

    game_completions = completions[completions['gameId'] == gid]
    game_plays_info = plays[plays['gameId'] == gid].set_index('playId')

    for _, play_row in game_completions.iterrows():
        pid = play_row['playId']
        target_id = int(play_row['targetNflId'])
        poss_team = play_row['possessionTeam']
        def_team = play_row['defensiveTeam']
        air_yds = play_row['air_yards']
        actual_yac = play_row['yards_after_catch']

        play_trk = trk[trk['playId'] == pid]
        if len(play_trk) == 0:
            continue

        # Snap frame
        snap_events = play_trk[play_trk['event'].isin(['ball_snap', 'autoevent_ballsnap'])]
        if len(snap_events) == 0:
            continue
        snap_frame = snap_events['frameId'].min()

        # Throw frame
        throw_events = play_trk[play_trk['event'].isin(['pass_forward', 'autoevent_passforward'])]
        if len(throw_events) == 0:
            continue
        throw_frame = throw_events['frameId'].min()

        # Ball at snap → LOS
        ball_snap = play_trk[(play_trk['frameId'] == snap_frame) & (play_trk['team'] == 'football')]
        if len(ball_snap) == 0:
            continue
        los_x = ball_snap.iloc[0]['x']

        # Target receiver at throw
        recv_throw = play_trk[(play_trk['frameId'] == throw_frame) & (play_trk['nflId'] == target_id)]
        if len(recv_throw) == 0:
            continue
        recv = recv_throw.iloc[0]

        # Defenders at throw
        def_throw = play_trk[(play_trk['frameId'] == throw_frame) & (play_trk['team'] == def_team)]
        def_throw = def_throw[def_throw['nflId'].notna()]
        if len(def_throw) == 0:
            continue

        # Catch point
        catch_x = los_x + air_yds
        # Flight time for catch_y estimate
        ball_throw = play_trk[(play_trk['frameId'] == throw_frame) & (play_trk['team'] == 'football')]
        if len(ball_throw) > 0:
            ball_at_throw_x = ball_throw.iloc[0]['x']
            ball_dist = abs(catch_x - ball_at_throw_x)
        else:
            ball_dist = abs(catch_x - recv['x'])
        t_f = np.clip(ball_dist / BALL_SPEED, MIN_FLIGHT_TIME, MAX_FLIGHT_TIME)
        catch_y = np.clip(recv['y'] + recv['vy'] * t_f, 0, 53.3)

        # Compute YAC available
        yac_avail, best_angle, end_x, end_y = compute_yac_available(
            recv['x'], recv['y'], recv['vx'], recv['vy'],
            def_throw['x'].values, def_throw['y'].values,
            def_throw['vx'].values, def_throw['vy'].values,
            catch_x, catch_y, los_x
        )

        results.append({
            'gameId': gid, 'playId': pid,
            'targetNflId': target_id,
            'los_x': los_x, 'catch_x': catch_x, 'catch_y': catch_y,
            'air_yards': air_yds, 'actual_yac': actual_yac,
            'yac_available': yac_avail, 'best_angle': best_angle,
            'end_x': end_x, 'end_y': end_y,
            'xyac_mean': play_row.get('xyac_mean_yardage', np.nan),
        })

    if (gi + 1) % 20 == 0 or gi == len(game_ids) - 1:
        print(f"  Processed {gi+1}/{len(game_ids)} games, {len(results)} plays so far …")

df = pd.DataFrame(results)
print(f"\n  Total plays with YAC available: {len(df)}")

# Sanity check
print(f"\n[5] Sanity check — YAC available distribution:")
print(f"  Mean:   {df['yac_available'].mean():.2f}")
print(f"  Median: {df['yac_available'].median():.2f}")
print(f"  Std:    {df['yac_available'].std():.2f}")
print(f"  Min:    {df['yac_available'].min():.2f}")
print(f"  Max:    {df['yac_available'].max():.2f}")
print(f"  Pct = 0: {(df['yac_available'] == 0).mean()*100:.1f}%")

# ── Merge player info ────────────────────────────────────────────────
df = df.merge(players[['nflId', 'displayName', 'officialPosition']],
              left_on='targetNflId', right_on='nflId', how='left')
df['yac_over_available'] = df['actual_yac'] - df['yac_available']

# ── VALIDATION ───────────────────────────────────────────────────────
print("\n[6] Validation …")
valid = df.dropna(subset=['actual_yac', 'yac_available'])

# Correlations
r_avail, p_avail = stats.pearsonr(valid['actual_yac'], valid['yac_available'])
rho_avail, _ = stats.spearmanr(valid['actual_yac'], valid['yac_available'])
print(f"  Pearson  corr(YAC, YAC_available)  = {r_avail:.3f} (p={p_avail:.2e})")
print(f"  Spearman corr(YAC, YAC_available)  = {rho_avail:.3f}")

valid_xyac = valid.dropna(subset=['xyac_mean'])
if len(valid_xyac) > 0:
    r_xyac, _ = stats.pearsonr(valid_xyac['actual_yac'], valid_xyac['xyac_mean'])
    rho_xyac, _ = stats.spearmanr(valid_xyac['actual_yac'], valid_xyac['xyac_mean'])
    print(f"  Pearson  corr(YAC, xYAC)           = {r_xyac:.3f}")
    print(f"  Spearman corr(YAC, xYAC)           = {rho_xyac:.3f}")

# OLS: YAC ~ xYAC vs YAC ~ xYAC + YAC_available
valid_ols = valid.dropna(subset=['xyac_mean'])
if len(valid_ols) > 50:
    y = valid_ols['actual_yac'].values.reshape(-1, 1)

    # Model 1: xYAC only
    X1 = valid_ols[['xyac_mean']].values
    m1 = LinearRegression().fit(X1, y)
    r2_xyac_only = m1.score(X1, y)

    # Model 2: xYAC + YAC_available
    X2 = valid_ols[['xyac_mean', 'yac_available']].values
    m2 = LinearRegression().fit(X2, y)
    r2_both = m2.score(X2, y)

    r2_gain = r2_both - r2_xyac_only

    print(f"\n  OLS R² — xYAC only:           {r2_xyac_only:.4f}")
    print(f"  OLS R² — xYAC + YAC_avail:    {r2_both:.4f}")
    print(f"  R² gain from YAC_available:    {r2_gain:.4f}")

    # Split by air yards
    for label, mask in [('short (air < 10)', valid_ols['air_yards'] < 10),
                        ('deep (air >= 10)', valid_ols['air_yards'] >= 10)]:
        sub = valid_ols[mask]
        if len(sub) > 30:
            ys = sub['actual_yac'].values.reshape(-1, 1)
            X1s = sub[['xyac_mean']].values
            X2s = sub[['xyac_mean', 'yac_available']].values
            r2_1 = LinearRegression().fit(X1s, ys).score(X1s, ys)
            r2_2 = LinearRegression().fit(X2s, ys).score(X2s, ys)
            print(f"  {label}: R²(xYAC)={r2_1:.4f}, R²(xYAC+avail)={r2_2:.4f}, gain={r2_2-r2_1:.4f}")

# ── LEADERBOARD (position-neutral residual) ──────────────────────────
print("\n[7] Building leaderboard …")

# Fit expected-YAC model on ALL plays (for play-level use & play field views)
leader_ols = df.dropna(subset=['xyac_mean', 'yac_available', 'air_yards', 'actual_yac']).copy()
X_leader = leader_ols[['yac_available', 'xyac_mean', 'air_yards']].values
y_leader = leader_ols['actual_yac'].values
expected_model = LinearRegression().fit(X_leader, y_leader)
leader_ols['predicted_yac'] = expected_model.predict(X_leader)
leader_ols['yac_over_expected'] = leader_ols['actual_yac'] - leader_ols['predicted_yac']
r2_expected = expected_model.score(X_leader, y_leader)
print(f"  Expected-YAC model R² (avail + xYAC + air_yards, all plays): {r2_expected:.4f}")

# Full df with predictions (for play field views)
df_for_leader = df.dropna(subset=['xyac_mean', 'yac_available', 'air_yards']).copy()
X_all = df_for_leader[['yac_available', 'xyac_mean', 'air_yards']].values
df_for_leader['predicted_yac'] = expected_model.predict(X_all)
df_for_leader['yac_over_expected'] = df_for_leader['actual_yac'] - df_for_leader['predicted_yac']

# ── SHORT-THROW player analysis (air_yards < 10) for quadrant & leaderboard ──
MIN_REC_SHORT = 10
short_df = df_for_leader[df_for_leader['air_yards'] < 10].copy()
print(f"\n  Short completions (air < 10): {len(short_df)}")

short_player = short_df.groupby(['targetNflId', 'displayName', 'officialPosition']).agg(
    n=('yac_over_expected', 'size'),
    mean_yac_over_expected=('yac_over_expected', 'mean'),
    mean_yac=('actual_yac', 'mean'),
    mean_yac_avail=('yac_available', 'mean'),
    mean_air_yards=('air_yards', 'mean'),
).reset_index()

qualified = short_player[short_player['n'] >= MIN_REC_SHORT].sort_values('mean_yac_over_expected', ascending=False)
print(f"  Qualified players (>={MIN_REC_SHORT} short receptions): {len(qualified)}")

# Check player-level correlation on short throws
player_r_short, player_p_short = stats.pearsonr(qualified['mean_yac_avail'], qualified['mean_yac'])
print(f"\n  Player-level Pearson corr(mean_YAC_avail, mean_YAC) [short]: {player_r_short:.3f} (p={player_p_short:.2e})")

# If corr <= 0, depth-adjust YAC_available by residualizing on air_yards bins
use_depth_adjusted = (player_r_short <= 0)
if use_depth_adjusted:
    print("  Corr <= 0 on short throws => depth-adjusting YAC_available …")
    # Bin air yards: <0, 0-4, 5-9
    def air_bin(ay):
        if ay < 0: return '<0'
        elif ay < 5: return '0-4'
        else: return '5-9'
    short_df['air_bin'] = short_df['air_yards'].apply(air_bin)
    bin_means = short_df.groupby('air_bin')['yac_available'].mean()
    print(f"  Bin means: {dict(bin_means.round(2))}")
    short_df['yac_avail_adj'] = short_df['yac_available'] - short_df['air_bin'].map(bin_means)

    short_player = short_df.groupby(['targetNflId', 'displayName', 'officialPosition']).agg(
        n=('yac_over_expected', 'size'),
        mean_yac_over_expected=('yac_over_expected', 'mean'),
        mean_yac=('actual_yac', 'mean'),
        mean_yac_avail=('yac_avail_adj', 'mean'),   # depth-adjusted
        mean_air_yards=('air_yards', 'mean'),
    ).reset_index()
    qualified = short_player[short_player['n'] >= MIN_REC_SHORT].sort_values('mean_yac_over_expected', ascending=False)

    player_r, player_p = stats.pearsonr(qualified['mean_yac_avail'], qualified['mean_yac'])
    print(f"  Depth-adjusted Pearson corr(adj_YAC_avail, mean_YAC) [short]: {player_r:.3f} (p={player_p:.2e})")
    quad_x_label = 'Open field vs. typical for throw depth (yds)'
else:
    player_r, player_p = player_r_short, player_p_short
    quad_x_label = 'Mean YAC Available (yards)'

print(f"\n  Top 5 (short throws):")
for _, r in qualified.head(5).iterrows():
    print(f"    {r['displayName']:25s} ({r['officialPosition']}) n={r['n']:3d}  "
          f"YAC/exp={r['mean_yac_over_expected']:+.1f}  YAC={r['mean_yac']:.1f}  avail={r['mean_yac_avail']:.1f}")
print(f"  Bottom 5 (short throws):")
for _, r in qualified.tail(5).iterrows():
    print(f"    {r['displayName']:25s} ({r['officialPosition']}) n={r['n']:3d}  "
          f"YAC/exp={r['mean_yac_over_expected']:+.1f}  YAC={r['mean_yac']:.1f}  avail={r['mean_yac_avail']:.1f}")

# Position mix check
pos_counts_top10 = qualified.head(10)['officialPosition'].value_counts()
pos_counts_bot10 = qualified.tail(10)['officialPosition'].value_counts()
print(f"\n  Position mix — Top 10: {dict(pos_counts_top10)}")
print(f"  Position mix — Bot 10: {dict(pos_counts_bot10)}")

# Save leaderboard CSV
qualified.to_csv(os.path.join(OUTPUT_DIR, 'player_leaderboard.csv'), index=False)

# ── PLOTS ────────────────────────────────────────────────────────────
print("\n[8] Generating plots …")

# Color map for positions
pos_colors = {'WR': '#1f77b4', 'TE': '#ff7f0e', 'RB': '#2ca02c', 'FB': '#d62728'}

# --- Plot 1 (HERO): Leaderboard (SHORT throws) — overall top/bottom 10 + position panels ---
fig = plt.figure(figsize=(22, 9))
gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 1, 1, 1])
fig.subplots_adjust(wspace=0.9)

ax0 = fig.add_subplot(gs[0])
top10 = qualified.head(10)
bot10 = qualified.tail(10)
overall = pd.concat([top10, bot10]).drop_duplicates().sort_values('mean_yac_over_expected', ascending=True)
colors_overall = [pos_colors.get(p, '#888') for p in overall['officialPosition']]
labels_overall = [f"{r['displayName']} ({r['officialPosition']}, n={r['n']})" for _, r in overall.iterrows()]
ax0.barh(range(len(overall)), overall['mean_yac_over_expected'].values,
         color=colors_overall, edgecolor='white', linewidth=0.5)
ax0.set_yticks(range(len(overall)))
ax0.set_yticklabels(labels_overall, fontsize=7.5)
ax0.axvline(0, color='black', linewidth=0.8)
ax0.set_xlabel('Mean YAC Over Expected (yd)', fontsize=9)
ax0.set_title('Overall Top/Bottom 10', fontsize=11, fontweight='bold')

for i, pos in enumerate(['WR', 'TE', 'RB']):
    ax = fig.add_subplot(gs[i + 1])
    pos_df = qualified[qualified['officialPosition'] == pos].copy()
    if len(pos_df) == 0:
        ax.set_title(pos)
        continue
    top5 = pos_df.head(5)
    bot5 = pos_df.tail(5)
    panel = pd.concat([top5, bot5]).drop_duplicates().sort_values('mean_yac_over_expected', ascending=True)
    color = pos_colors.get(pos, '#888')
    labels = [f"{r['displayName']} (n={r['n']})" for _, r in panel.iterrows()]
    ax.barh(range(len(panel)), panel['mean_yac_over_expected'].values,
            color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.set_yticks(range(len(panel)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Mean YAC Over Expected (yd)', fontsize=9)
    ax.set_title(pos, fontsize=11, fontweight='bold')

# Legend: only positions actually present (no FB)
present_positions = [p for p in ['WR', 'TE', 'RB'] if p in qualified['officialPosition'].values]
patches_legend = [mpatches.Patch(color=pos_colors[p], label=p) for p in present_positions]
ax0.legend(handles=patches_legend, loc='lower right', fontsize=8)

fig.suptitle(f'YAC Over Expected on Short Throws (2021 Wks 1-8, min {MIN_REC_SHORT} rec)\n'
             'Actual YAC minus model-expected YAC (tracking open field + nflverse xYAC + air yards), air yards < 10',
             fontsize=13, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, 'leaderboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved leaderboard.png")

# --- Delete old play PNGs ---
import glob
for old_play in glob.glob(os.path.join(OUTPUT_DIR, 'play_*.png')):
    os.remove(old_play)
    print(f"  Deleted {os.path.basename(old_play)}")

# --- Plot 3: Single play field view (biggest positive, excluding sideline catches) ---
# Filter: catch_y must be > 1 and < 52.3 (at least 1 yd from sideline)
eligible = df_for_leader[(df_for_leader['catch_y'] > 1.0) & (df_for_leader['catch_y'] < 52.3)]
biggest_pos = eligible.loc[eligible['yac_over_expected'].idxmax()]

def plot_play_field(play_info, filename):
    """Draw field view for a single play — clean style."""
    gid = int(play_info['gameId'])
    pid = int(play_info['playId'])

    fname = os.path.join(TRACKING_DIR, f'tracking_{gid}.csv')
    trk = pd.read_csv(fname, usecols=tracking_cols)
    trk = normalize_tracking(trk)
    trk = add_velocity(trk)

    play_trk = trk[trk['playId'] == pid]
    play_info_row = plays[(plays['gameId'] == gid) & (plays['playId'] == pid)].iloc[0]
    def_team = play_info_row['defensiveTeam']

    throw_events = play_trk[play_trk['event'].isin(['pass_forward', 'autoevent_passforward'])]
    throw_frame = throw_events['frameId'].min()

    def_throw = play_trk[(play_trk['frameId'] == throw_frame) & (play_trk['team'] == def_team)]
    def_throw = def_throw[def_throw['nflId'].notna()]

    catch_x = play_info['catch_x']
    catch_y = play_info['catch_y']
    end_x = play_info['end_x']
    end_y = play_info['end_y']
    los_x = play_info['los_x']

    ball_dist = max(abs(catch_x - play_trk[(play_trk['frameId'] == throw_frame) & (play_trk['team'] == 'football')].iloc[0]['x']), 1.0)
    t_f = np.clip(ball_dist / BALL_SPEED, MIN_FLIGHT_TIME, MAX_FLIGHT_TIME)
    proj_def_x, proj_def_y = project_defenders_break(
        def_throw['x'].values, def_throw['y'].values, catch_x, catch_y, t_f)

    # Grid covers full field width, reasonable x range around catch
    grid_x_range = np.arange(max(catch_x - 5, 0), min(catch_x + 40, 120), 0.5)
    grid_y_range = np.arange(0, 53.3, 0.5)
    gx, gy = np.meshgrid(grid_x_range, grid_y_range)

    recv_dists = np.sqrt((gx - catch_x)**2 + (gy - catch_y)**2)
    recv_times = RECEIVER_CATCH_DELAY + recv_dists / RECEIVER_SPEED

    min_def_times = np.full_like(gx, np.inf)
    for dx, dy in zip(proj_def_x, proj_def_y):
        d_dists = np.sqrt((gx - dx)**2 + (gy - dy)**2)
        effective_dists = np.maximum(d_dists - TACKLE_RADIUS, 0.0)
        d_times = DEFENDER_REACTION + effective_dists / DEFENDER_SPEED
        min_def_times = np.minimum(min_def_times, d_times)

    open_field = recv_times < min_def_times

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('#f5f5f5')

    # Yard lines
    for yl in range(0, 121, 10):
        ax.axvline(yl, color='#ddd', linewidth=0.5)
    # Sidelines
    ax.axhline(0, color='#aaa', linewidth=1)
    ax.axhline(53.3, color='#aaa', linewidth=1)
    # LOS
    ax.axvline(los_x, color='black', linewidth=1.5, linestyle='--', label='Line of scrimmage')
    # End zone
    ax.axvspan(110, 120, color='#ffe0b2', alpha=0.4)

    # Green band along the arrow (model's open lane, ~2 yd wide)
    lane_dx = end_x - catch_x
    lane_dy = end_y - catch_y
    lane_len = np.sqrt(lane_dx**2 + lane_dy**2)
    if lane_len > 0:
        perp_x = -lane_dy / lane_len
        perp_y = lane_dx / lane_len
    else:
        perp_x, perp_y = 0, 1
    band_hw = 1.0  # half-width in yards
    band_poly = plt.Polygon([
        (catch_x + band_hw * perp_x, catch_y + band_hw * perp_y),
        (end_x + band_hw * perp_x, end_y + band_hw * perp_y),
        (end_x - band_hw * perp_x, end_y - band_hw * perp_y),
        (catch_x - band_hw * perp_x, catch_y - band_hw * perp_y),
    ], closed=True, color='#81c784', alpha=0.4, zorder=3,
       label=f"Model's open lane (~{play_info['yac_available']:.0f} yds)")
    ax.add_patch(band_poly)

    # Defenders
    ax.scatter(proj_def_x, proj_def_y, c='red', s=90, marker='x', linewidths=2.5,
               zorder=5, label='Defenders (projected)')
    # Receiver at catch point
    ax.scatter(catch_x, catch_y, c='#1565c0', s=130, marker='o', zorder=6,
               edgecolors='white', linewidths=2, label='Receiver (catch point)')
    # Best lane arrow
    ax.annotate('', xy=(end_x, end_y), xytext=(catch_x, catch_y),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=3.5), zorder=7)

    ax.set_xlim(max(catch_x - 10, 0), min(catch_x + 42, 120))
    ax.set_ylim(0, 53.3)
    ax.set_xlabel('Field x (yards from left end zone)', fontsize=10)
    ax.set_ylabel('Field y (yards)', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    yac_avail = play_info['yac_available']
    actual = play_info['actual_yac']
    player_name = play_info.get('displayName', 'Player')
    ax.set_title(f"{player_name}: model saw ~{yac_avail:.0f} yds of open field, he gained {actual:.0f}",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

plot_play_field(biggest_pos, "play_chase.png")



elapsed = time.time() - t0
print(f"\n=== Done in {elapsed:.1f}s ===")

# ── Store key numbers for README ─────────────────────────────────────
print("\n-- README Numbers --")
print(f"TOTAL_PLAYS={len(df)}")
print(f"R2_XYAC={r2_xyac_only:.4f}")
print(f"R2_BOTH={r2_both:.4f}")
print(f"R2_GAIN={r2_gain:.4f}")
print(f"R2_EXPECTED_MODEL={r2_expected:.4f}")
print(f"PEARSON_AVAIL={r_avail:.3f}")
print(f"PLAYER_PEARSON_SHORT={player_r:.3f}")
print(f"DEPTH_ADJUSTED={use_depth_adjusted}")
print(f"MEDIAN_YAC_AVAIL={df['yac_available'].median():.1f}")
print(f"ACTUAL_YAC_MEDIAN={df['actual_yac'].median():.1f}")
print(f"SHORT_PLAYS={len(short_df)}")
print(f"SHORT_QUALIFIED={len(qualified)}")
if len(qualified) > 0:
    top1 = qualified.iloc[0]
    bot1 = qualified.iloc[-1]
    print(f"TOP_PLAYER={top1['displayName']} ({top1['officialPosition']}) +{top1['mean_yac_over_expected']:.1f}")
    print(f"BOT_PLAYER={bot1['displayName']} ({bot1['officialPosition']}) {bot1['mean_yac_over_expected']:.1f}")
