"""
SUCROSE PREFERENCE TEST BEHAVIORAL ANALYSIS FOR DROSOPHILA MELANOGASTER
This code processes centroid tracking data from Bonsai.rx to extract behavioral parameters relevants for SPT such as:
- Time at each extremes
- Preference index
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Parameters

fps_default = 14.0           # Default FPS if exact value is not available
mm_ref = 50.0                # Real reference distance (mm) - arena length
min_distance = 2.0           # Minimum distance for occupancy detection (mm)
min_duration = 2.0           # Minimum duration for occupancy detection (s)

# Analysis time interval in seconds (None = no limit)
time_interval_s = (0, 600)   # Example: from 0s to 600s (entire video)

# Root directory containing experimental data
root_dir = r"your\folder"

#1D Metrics functions

def total_distance(df):
    """
    Calculate total distance traveled in 1D.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'X_mm' column
    
    Returns
    -------
    float
        Total distance traveled in mm
    """
    if "X_mm" not in df.columns:
        raise KeyError("Missing 'X_mm' column")
    dx = df["X_mm"].diff()
    return float(np.nansum(np.abs(dx)))


def average_distance_to_extremes(df, arena_length_mm):
    """
    Calculate average distance to each extreme (0 and arena_length_mm).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'X_mm' column
    arena_length_mm : float
        Total length of arena in mm
    
    Returns
    -------
    tuple of float
        (avg_distance_to_left, avg_distance_to_right) in mm
    """
    if "X_mm" not in df.columns:
        raise KeyError("Missing 'X_mm' column")
    dist_left = df["X_mm"]                      # distance to left extreme (0)
    dist_right = arena_length_mm - df["X_mm"]  # distance to right extreme
    return float(dist_left.mean()), float(dist_right.mean())


def time_in_half(df, arena_length_mm, fps):
    """
    Calculate time spent in each half of the arena.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'X_mm' column
    arena_length_mm : float
        Total length of arena in mm
    fps : float
        Frames per second
    
    Returns
    -------
    tuple of float
        (time_left_half, time_right_half) in seconds
    """
    if "X_mm" not in df.columns:
        raise KeyError("Missing 'X_mm' column")
    midpoint = arena_length_mm / 2.0
    time_left = (df["X_mm"] <= midpoint).sum() / fps
    time_right = (df["X_mm"] > midpoint).sum() / fps
    return float(time_left), float(time_right)


def _calculate_extreme_episodes(df, extreme, arena_length_mm, fps, min_dist, min_dur):
    """
    Private helper function to calculate valid episodes at an extreme.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'X_mm' column
    extreme : str
        'left' or 'right'
    arena_length_mm : float
        Total length of arena in mm
    fps : float
        Frames per second
    min_dist : float
        Minimum distance to extreme for occupancy (mm)
    min_dur : float
        Minimum duration for valid occupancy (s)
    
    Returns
    -------
    list of tuple
        List of (start_idx, end_idx, duration_frames) for valid episodes
    """
    if "X_mm" not in df.columns:
        raise KeyError("Missing 'X_mm' column")
    
    if extreme == "left":
        in_extreme = df["X_mm"] <= min_dist
    elif extreme == "right":
        in_extreme = df["X_mm"] >= (arena_length_mm - min_dist)
    else:
        raise ValueError("extreme must be 'left' or 'right'")
    
    # Detect entry points
    changes = in_extreme.astype(int).diff().fillna(0)
    starts = np.where(changes == 1)[0]
    
    valid_episodes = []
    for start in starts:
        # Find exit point
        exits = np.where(~in_extreme.iloc[start:])[0]
        end = start + exits[0] if len(exits) > 0 else len(df)
        dur_frames = end - start
        
        # Check if episode meets minimum duration
        if dur_frames >= fps * min_dur:
            valid_episodes.append((start, end, dur_frames))
    
    return valid_episodes


def time_in_extreme(df, extreme="left", arena_length_mm=mm_ref, fps=fps_default,
                    min_dist=min_distance, min_dur=min_duration):
    """
    Calculate total time spent in an extreme, considering minimum distance and duration.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'X_mm' column
    extreme : str
        'left' or 'right'
    arena_length_mm : float
        Total length of arena in mm
    fps : float
        Frames per second
    min_dist : float
        Minimum distance to extreme for occupancy (mm)
    min_dur : float
        Minimum duration for valid occupancy (s)
    
    Returns
    -------
    float
        Total time in extreme in seconds
    """
    episodes = _calculate_extreme_episodes(df, extreme, arena_length_mm, fps, min_dist, min_dur)
    total_time = sum(ep[2] for ep in episodes) / fps
    return float(total_time)


def entries_in_extreme(df, extreme="left", arena_length_mm=mm_ref, fps=fps_default,
                       min_dist=min_distance, min_dur=min_duration):
    """
    Count number of valid entries to an extreme.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'X_mm' column
    extreme : str
        'left' or 'right'
    arena_length_mm : float
        Total length of arena in mm
    fps : float
        Frames per second
    min_dist : float
        Minimum distance to extreme for occupancy (mm)
    min_dur : float
        Minimum duration for valid occupancy (s)
    
    Returns
    -------
    int
        Number of valid entries
    """
    episodes = _calculate_extreme_episodes(df, extreme, arena_length_mm, fps, min_dist, min_dur)
    return len(episodes)


def preference_index_halves(time_left, time_right):
    """
    Calculate preference index between halves.
    
    Formula: (time_right - time_left) / (time_right + time_left)
    Range: [-1, 1] where -1 = complete left preference, +1 = complete right preference
    
    Parameters
    ----------
    time_left : float
        Time spent in left half (s)
    time_right : float
        Time spent in right half (s)
    
    Returns
    -------
    float
        Preference index or np.nan if total time is 0
    """
    total = time_left + time_right
    if total == 0:
        return np.nan
    return (time_right - time_left) / total


def preference_index_extremes(time_ext_left, time_ext_right):
    """
    Calculate preference index between extremes.
    
    Formula: (time_ext_right - time_ext_left) / (time_ext_right + time_ext_left)
    Range: [-1, 1] where -1 = complete left preference, +1 = complete right preference
    
    Parameters
    ----------
    time_ext_left : float
        Time spent in left extreme (s)
    time_ext_right : float
        Time spent in right extreme (s)
    
    Returns
    -------
    float
        Preference index or np.nan if total time is 0
    """
    total = time_ext_left + time_ext_right
    if total == 0:
        return np.nan
    return (time_ext_right - time_ext_left) / total


def average_time_per_entry(time_extreme, entries):
    """
    Calculate average time per entry to an extreme.
    
    Parameters
    ----------
    time_extreme : float
        Total time in extreme (s)
    entries : int
        Number of entries
    
    Returns
    -------
    float
        Average time per entry in seconds, or np.nan if no entries
    """
    if entries == 0:
        return np.nan
    return time_extreme / entries

def adjust_metrics_for_sugar_side(metrics, sugar_side):
    """
    Adjust metrics so that 'sugar' side always refers to the side with sucrose.
    
    Renames left/right metrics to sugar/control based on which side has sugar.
    For preference indices, adjusts sign so positive always means sugar preference.
    
    Parameters
    ----------
    metrics : dict
        Original metrics from process_track_1d
    sugar_side : str
        'left' or 'right' - which side has the sugar
    
    Returns
    -------
    dict
        Adjusted metrics with sugar/control nomenclature
        
    Examples
    --------
    If sugar_side='left':
        - time_extreme_left becomes time_extreme_sugar
        - time_extreme_right becomes time_extreme_control
        - preference_index_extremes sign is flipped
    """
    adjusted = {}
    
    # Total distance doesn't depend on side
    adjusted['total_distance'] = metrics['total_distance']
    
    if sugar_side == 'left':
        # Sugar is on LEFT → LEFT = SUGAR, RIGHT = CONTROL
        adjusted['avg_distance_sugar'] = metrics['avg_distance_left']
        adjusted['avg_distance_control'] = metrics['avg_distance_right']
        adjusted['time_half_sugar'] = metrics['time_half_left']
        adjusted['time_half_control'] = metrics['time_half_right']
        adjusted['time_extreme_sugar'] = metrics['time_extreme_left']
        adjusted['time_extreme_control'] = metrics['time_extreme_right']
        adjusted['entries_extreme_sugar'] = metrics['entries_extreme_left']
        adjusted['entries_extreme_control'] = metrics['entries_extreme_right']
        adjusted['avg_time_per_entry_sugar'] = metrics['avg_time_per_entry_left']
        adjusted['avg_time_per_entry_control'] = metrics['avg_time_per_entry_right']
        
        # Flip preference indices (positive should mean sugar preference)
        adjusted['preference_index_halves'] = -metrics['preference_index_halves']
        adjusted['preference_index_extremes'] = -metrics['preference_index_extremes']
        
    elif sugar_side == 'right':
        # Sugar is on RIGHT → RIGHT = SUGAR, LEFT = CONTROL
        adjusted['avg_distance_sugar'] = metrics['avg_distance_right']
        adjusted['avg_distance_control'] = metrics['avg_distance_left']
        adjusted['time_half_sugar'] = metrics['time_half_right']
        adjusted['time_half_control'] = metrics['time_half_left']
        adjusted['time_extreme_sugar'] = metrics['time_extreme_right']
        adjusted['time_extreme_control'] = metrics['time_extreme_left']
        adjusted['entries_extreme_sugar'] = metrics['entries_extreme_right']
        adjusted['entries_extreme_control'] = metrics['entries_extreme_left']
        adjusted['avg_time_per_entry_sugar'] = metrics['avg_time_per_entry_right']
        adjusted['avg_time_per_entry_control'] = metrics['avg_time_per_entry_left']
        
        # Preference indices stay the same (positive already means right=sugar preference)
        adjusted['preference_index_halves'] = metrics['preference_index_halves']
        adjusted['preference_index_extremes'] = metrics['preference_index_extremes']
    
    else:
        raise ValueError(f"Invalid sugar_side: '{sugar_side}'. Must be 'left' or 'right'")
    
    return adjusted

def analyze_experiments_1d(
    root_dir,
    fps=fps_default,
    video_duration_s=600,
    output_dir="results",
    time_interval_s=(None, None),
    mm_ref=mm_ref,
    min_dist=min_distance,
    min_dur=min_duration,
):
    """
    Process all 1D tracking experiments in a root directory and summarize metrics.
    
    Traverses the experimental directory structure, processes each track, and exports
    results to CSV files (one per condition). Automatically adjusts metrics based on
    which side has sugar (specified in config.csv).
    
    Expected directory structure:
    root_dir/
    ├── condition1/
    │   ├── N1/
    │   │   ├── largo.csv    # Arena length in pixels (single value)
    │   │   ├── config.csv   # Sugar side: 'left' or 'right' (single value)
    │   │   ├── fly1.csv     # X coordinates in pixels (single column, no header)
    │   │   └── fly2.csv
    │   └── N2/
    │       └── ...
    └── condition2/
        └── ...
    
    Parameters
    ----------
    root_dir : str
        Root directory containing experimental conditions
    fps : float, optional
        Frames per second (default: fps_default)
    video_duration_s : float, optional
        Total video duration in seconds (default: 600)
    output_dir : str, optional
        Directory where results will be saved (default: "results")
    time_interval_s : tuple, optional
        (start, end) time interval in seconds for analysis (default: (None, None))
    mm_ref : float, optional
        Real arena length in mm (default: mm_ref global)
    min_dist : float, optional
        Minimum distance to extreme for occupancy (mm) (default: min_distance global)
    min_dur : float, optional
        Minimum duration for occupancy (s) (default: min_duration global)
    
    Returns
    -------
    df_summary : pd.DataFrame
        DataFrame with all metrics summarized by track (with sugar/control nomenclature)
    dfs_stored : dict
        Dictionary with processed DataFrames, indexed by (condition, N, file)
    
    Notes
    -----
    Files are expected to follow naming convention: fly*.csv
    largo.csv must contain a single numeric value (arena length in pixels)
    config.csv must contain 'left' or 'right' (with optional 'sugar_side' header)
    """
    results = []
    dfs_stored = {}
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Analyzing experiments in: {root_dir}\n")

    # Iterate through conditions
    for condition in sorted(os.listdir(root_dir)):
        path_cond = os.path.join(root_dir, condition)
        if not os.path.isdir(path_cond):
            continue

        print(f"🧪 Condition: {condition}")

        # Iterate through replicates (N)
        for N in sorted(os.listdir(path_cond)):
            path_N = os.path.join(path_cond, N)
            if not os.path.isdir(path_N):
                continue

            # Read arena length from largo.csv
            largo_path = os.path.join(path_N, "largo.csv")
            if not os.path.exists(largo_path):
                print(f"  ⚠️  largo.csv not found in {path_N}, skipping.")
                continue
            
            try:
                arena_length_px = float(pd.read_csv(largo_path, header=None).iloc[0, 0])
                if arena_length_px <= 0:
                    raise ValueError(f"Invalid arena length: {arena_length_px}")
            except (ValueError, IndexError, pd.errors.EmptyDataError) as e:
                print(f"  ❌ Error reading largo.csv in {path_N}: {e}")
                continue
            except Exception as e:
                print(f"  ❌ Unexpected error reading largo.csv in {path_N}: {type(e).__name__}: {e}")
                continue

            # ========== READ SUGAR SIDE CONFIGURATION ==========
            config_path = os.path.join(path_N, "config.csv")
            sugar_side = 'right'  # Default
            
            if os.path.exists(config_path):
                try:
                    config_df = pd.read_csv(config_path, header=None)
                    sugar_side_raw = str(config_df.iloc[0, 0]).strip().lower()
                    
                    # Handle potential header
                    if sugar_side_raw == 'sugar_side' and len(config_df) > 1:
                        sugar_side_raw = str(config_df.iloc[1, 0]).strip().lower()
                    
                    if sugar_side_raw in ['left', 'right']:
                        sugar_side = sugar_side_raw
                        print(f"  ✓ Sugar side: {sugar_side}")
                    else:
                        print(f"  ⚠️  Invalid sugar_side '{sugar_side_raw}' in config.csv, using default 'right'")
                        
                except Exception as e:
                    print(f"  ⚠️  Error reading config.csv in {path_N}: {e}")
                    print(f"  ⚠️  Using default sugar_side='right'")
            else:
                print(f"  ⚠️  config.csv not found in {path_N}, using default sugar_side='right'")
            # ===================================================

            # Process all fly*.csv files
            fly_files = [f for f in sorted(os.listdir(path_N)) 
                        if f.startswith("fly") and f.endswith(".csv")]
            
            if not fly_files:
                print(f"  ⚠️  No fly*.csv files found in {path_N}")
                continue

            for filename in fly_files:
                filepath = os.path.join(path_N, filename)

                try:
                    # Process 1D track
                    df, metrics = process_track_1d(
                        filepath,
                        arena_length_px=arena_length_px,
                        video_duration_s=video_duration_s,
                        mm_ref=mm_ref,
                        fps=fps,
                        time_interval_s=time_interval_s,
                        min_dist=min_dist,
                        min_dur=min_dur,
                    )

                    # ========== ADJUST METRICS BASED ON SUGAR SIDE ==========
                    metrics_adjusted = adjust_metrics_for_sugar_side(metrics, sugar_side)
                    # ========================================================

                    # Add metadata
                    metrics_adjusted.update({
                        "condition": condition,
                        "N": N,
                        "file": filename,
                        "sugar_side": sugar_side  # Store for reference
                    })
                    results.append(metrics_adjusted)

                    # Store processed DataFrame
                    dfs_stored[(condition, N, filename)] = df

                    print(f"    ✓ {filename}")

                except FileNotFoundError as e:
                    print(f"  ❌ File not found: {condition}/{N}/{filename}")
                    continue
                except (ValueError, KeyError) as e:
                    print(f"  ❌ Error processing {condition}/{N}/{filename}: {e}")
                    continue
                except Exception as e:
                    print(f"  ❌ Unexpected error in {condition}/{N}/{filename}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Generate summary
    if not results:
        print("\n⚠️  No results generated. Check paths and directory structure.")
        return pd.DataFrame(), {}

    df_summary = pd.DataFrame(results)
    
    print(f"\n📊 Summary: {len(results)} tracks processed from {len(df_summary['condition'].unique())} condition(s)")

    # Export results to CSV files
    export_results_to_csv(df_summary, output_dir)

    print(f"\n✅ Analysis complete. Results saved to '{output_dir}/'")

    return df_summary, dfs_stored

def export_results_to_csv(df_summary, output_dir="results"):
    """
    Export df_summary to CSV files, one per condition, with ordered columns.
    
    Creates a directory with one CSV file per experimental condition.
    If 'condition' column is missing, exports a single 'results.csv' file.
    
    Parameters
    ----------
    df_summary : pd.DataFrame
        Summary DataFrame with all metrics and a 'condition' column
    output_dir : str, optional
        Directory where CSV files will be saved (default: "results")
    
    Raises
    ------
    ValueError
        If df_summary is empty or None
    
    Notes
    -----
    CSV files are named using the condition name (sanitized for filesystem).
    Columns use sugar/control nomenclature for clarity in preference tests.
    """
    if df_summary is None or len(df_summary) == 0:
        raise ValueError("df_summary is empty or None")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If no 'condition' column, create a single file
    if "condition" not in df_summary.columns:
        df_summary = df_summary.copy()
        df_summary["condition"] = "results"
    
    # Preserve original order of conditions
    conditions_in_order = df_summary["condition"].astype(object).drop_duplicates().tolist()
    
    # Define column order with sugar/control nomenclature
    column_order = [
        "N", "file", "sugar_side",
        "total_distance",
        "avg_distance_sugar",
        "avg_distance_control",
        "time_half_sugar",
        "time_half_control",
        "time_extreme_sugar",
        "time_extreme_control",
        "entries_extreme_sugar",
        "entries_extreme_control",
        "preference_index_halves",
        "preference_index_extremes",
        "avg_time_per_entry_sugar",
        "avg_time_per_entry_control",
    ]
    
    # Add any remaining columns (in case there are extra metrics)
    remaining_cols = [c for c in df_summary.columns if c not in column_order + ["condition"]]
    column_order += remaining_cols
    
    # Keep only existing columns
    column_order = [c for c in column_order if c in df_summary.columns]
    
    # Export one CSV per condition
    exported_files = []
    for cond in conditions_in_order:
        df_cond = df_summary[df_summary["condition"] == cond].copy()
        
        # Drop condition column from output
        if "condition" in df_cond.columns:
            df_cond = df_cond.drop(columns=["condition"])
        
        # Reorder columns
        cols_present = [c for c in column_order if c in df_cond.columns]
        df_cond = df_cond[cols_present]
        
        # Sanitize filename (remove invalid characters)
        safe_cond_name = str(cond).replace("/", "_").replace("\\", "_").replace(":", "_")
        filename = f"{safe_cond_name}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Export to CSV
        df_cond.to_csv(filepath, index=False, float_format='%.4f')
        exported_files.append(filepath)
        print(f"  ✓ Exported: {filename} ({len(df_cond)} rows)")
    
    print(f"\n✅ {len(exported_files)} CSV file(s) exported to '{output_dir}/'")
    return exported_files

def analyze_experiments_1d(
    root_dir,
    fps=fps_default,
    video_duration_s=600,
    output_dir="results",
    time_interval_s=(None, None),
    mm_ref=mm_ref,
    min_dist=min_distance,
    min_dur=min_duration,
):
    """
    Process all 1D tracking experiments in a root directory and summarize metrics.
    
    Traverses the experimental directory structure, processes each track, and exports
    results to CSV files (one per condition).
    
    Expected directory structure:
    root_dir/
    ├── condition1/
    │   ├── N1/
    │   │   ├── largo.csv    # Arena length in pixels (single value)
    │   │   ├── fly1.csv     # X coordinates in pixels (single column, no header)
    │   │   └── fly2.csv
    │   └── N2/
    │       └── ...
    └── condition2/
        └── ...
    
    Parameters
    ----------
    root_dir : str
        Root directory containing experimental conditions
    fps : float, optional
        Frames per second (default: fps_default)
    video_duration_s : float, optional
        Total video duration in seconds (default: 600)
    output_dir : str, optional
        Directory where results will be saved (default: "results")
    time_interval_s : tuple, optional
        (start, end) time interval in seconds for analysis (default: (None, None))
    mm_ref : float, optional
        Real arena length in mm (default: mm_ref global)
    min_dist : float, optional
        Minimum distance to extreme for occupancy (mm) (default: min_distance global)
    min_dur : float, optional
        Minimum duration for occupancy (s) (default: min_duration global)
    
    Returns
    -------
    df_summary : pd.DataFrame
        DataFrame with all metrics summarized by track
    dfs_stored : dict
        Dictionary with processed DataFrames, indexed by (condition, N, file)
    
    Notes
    -----
    Files are expected to follow naming convention: fly*.csv
    largo.csv must contain a single numeric value (arena length in pixels)
    """
    results = []
    dfs_stored = {}
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Analyzing experiments in: {root_dir}\n")

    # Iterate through conditions
    for condition in sorted(os.listdir(root_dir)):
        path_cond = os.path.join(root_dir, condition)
        if not os.path.isdir(path_cond):
            continue

        print(f"🧪 Condition: {condition}")

        # Iterate through replicates (N)
        for N in sorted(os.listdir(path_cond)):
            path_N = os.path.join(path_cond, N)
            if not os.path.isdir(path_N):
                continue

            # Read arena length from largo.csv
            largo_path = os.path.join(path_N, "largo.csv")
            if not os.path.exists(largo_path):
                print(f"  ⚠️  largo.csv not found in {path_N}, skipping.")
                continue
            
            try:
                arena_length_px = float(pd.read_csv(largo_path, header=None).iloc[0, 0])
                if arena_length_px <= 0:
                    raise ValueError(f"Invalid arena length: {arena_length_px}")
            except (ValueError, IndexError, pd.errors.EmptyDataError) as e:
                print(f"  ❌ Error reading largo.csv in {path_N}: {e}")
                continue
            except Exception as e:
                print(f"  ❌ Unexpected error reading largo.csv in {path_N}: {type(e).__name__}: {e}")
                continue

            # Process all fly*.csv files
            fly_files = [f for f in sorted(os.listdir(path_N)) 
                        if f.startswith("fly") and f.endswith(".csv")]
            
            if not fly_files:
                print(f"  ⚠️  No fly*.csv files found in {path_N}")
                continue

            for filename in fly_files:
                filepath = os.path.join(path_N, filename)

                try:
                    # Process 1D track
                    df, metrics = process_track_1d(
                        filepath,
                        arena_length_px=arena_length_px,
                        video_duration_s=video_duration_s,
                        mm_ref=mm_ref,
                        fps=fps,
                        time_interval_s=time_interval_s,
                        min_dist=min_dist,
                        min_dur=min_dur,
                    )

                    # Add metadata
                    metrics.update({
                        "condition": condition,
                        "N": N,
                        "file": filename
                    })
                    results.append(metrics)

                    # Store processed DataFrame
                    dfs_stored[(condition, N, filename)] = df

                    print(f"    ✓ Processed: {filename}")

                except FileNotFoundError as e:
                    print(f"  ❌ File not found: {condition}/{N}/{filename}")
                    continue
                except (ValueError, KeyError) as e:
                    print(f"  ❌ Error processing {condition}/{N}/{filename}: {e}")
                    continue
                except Exception as e:
                    print(f"  ❌ Unexpected error in {condition}/{N}/{filename}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Generate summary
    if not results:
        print("\n⚠️  No results generated. Check paths and directory structure.")
        return pd.DataFrame(), {}

    df_summary = pd.DataFrame(results)
    
    print(f"\n📊 Summary: {len(results)} tracks processed from {len(df_summary['condition'].unique())} condition(s)")

    # Export results to CSV files
    export_results_to_csv(df_summary, output_dir)

    print(f"\n✅ Analysis complete. Results saved to '{output_dir}/'")

    return df_summary, dfs_stored

# ---------------------------
# EXECUTION
# ---------------------------
if __name__ == "__main__":
    df_summary, dfs_stored = analyze_experiments_1d(
        root_dir=root_dir,
        time_interval_s=time_interval_s,
        fps=fps_default,
        video_duration_s=600,      # Adjust to your actual video duration
        output_dir="results"
    )