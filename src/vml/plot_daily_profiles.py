#!/usr/bin/env python3
"""
plot_daily_profiles.py
----------------------

Plot 24-hour ground-temperature profiles for a configurable subset of
stations, colouring each profile by how cold or warm it is.

USAGE
=====
    python plot_daily_profiles.py YYYYMMDD
                                  [--num N]
                                  [--stations ID [ID ...]]
                                  [--data-path PATH]
                                  [--save-dir DIR]
                                  [--no-show]

ARGUMENTS
---------
YYYYMMDD          Date to plot (e.g. 20240301)

OPTIONS
-------
--num/-n N        Number of random stations to plot (default 3).
--stations        Explicit list of station_id values to plot
                  (overrides --num if supplied).
--data-path       Directory with road_temp_YYYYMMDD.parquet
                  (default: /data/projects/glatmodel/obs/fild8/road_temp_daily).
--save-dir        Where to save PNG files (default: current directory).
--no-show         Do not open the figures; just save them.

EXAMPLES
--------
# plot three random stations
python plot_daily_profiles.py 20240301 --num 3

# plot two particular stations
python plot_daily_profiles.py 20240301 --stations 0-100001-0 0-100002-0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd

def plot_temperature_all(df: pd.DataFrame, date_str: str, station: str,
                         save_dir: Path, show: bool = True) -> None:
    """
    Plot all 24 hourly profiles for a single station on one figure.
    Colours correspond to hour of day, using discrete bins.
    """
    station_df = df[df.station_id == station].sort_values("timestamp")
    depths_cm = -np.arange(14, -1, -1)

    # Discrete colormap: 24 colors, one per hour
    cmap = plt.get_cmap("viridis", 24)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(0, 25), ncolors=24)

    fig, ax = plt.subplots(figsize=(8, 7))

    for _, row in station_df.iterrows():
        hour = pd.to_datetime(row["timestamp"]).hour
        color = cmap(norm(hour))
        temps = [row[f"depth_{i}"] for i in range(14, -1, -1)]
        temps_deg = [ t - 273.15 for t in temps]
        ax.plot(temps_deg, depths_cm, marker="o", lw=2, color=color)
    station_str = station.split("-")[1]
    ax.set(
        xlabel="Temperature (°C)",
        ylabel="Layer depth (cm)",
        title=f"Temperature profiles for {date_str} — {station_str}",
        ylim=(-15, 0),
    )
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.minorticks_on()

    # Discrete colorbar for hour of day
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=range(0, 24, 2), boundaries=np.arange(0, 25))
    cbar.set_label("Hour of Day")
    cbar.set_ticks(range(0, 24, 2))
    cbar.set_ticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])

    out_name = save_dir / f"profiles_{date_str}_{station.split('-')[1]}.png"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot daily ground-temperature profiles (color = hour of day, discrete).")
    parser.add_argument("date", help="Date string in YYYYMMDD format")
    parser.add_argument("--num", "-n", type=int, default=3,
                        help="How many random stations to plot")
    parser.add_argument("--stations", nargs="+",
                        help="Explicit list of station_id values to plot")
    parser.add_argument("--data-path", default="/data/projects/glatmodel/obs/fild8/road_profiles_daily",
                        help="Directory that holds road_temp_YYYYMMDD.parquet")
    parser.add_argument("--save-dir", default=".",
                        help="Directory to write PNG files")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display the figures (still saved)")

    args = parser.parse_args()
    date_str = args.date
    parquet_file = Path(args.data_path) / f"road_temp_{date_str}.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(f"Cannot find {parquet_file}")

    df = pd.read_parquet(parquet_file)
    stations_available = sorted(df["station_id"].unique())
    print(f"{len(stations_available)} station(s) available on {date_str}")

    # Decide which stations to plot
    if args.stations:
        chosen = [s for s in args.stations if s in stations_available]
        missing = set(args.stations) - set(chosen)
        if missing:
            print("WARNING  — station(s) not present and skipped:", ", ".join(missing))
    else:
        chosen = list(np.random.choice(stations_available,
                                       size=min(args.num, len(stations_available)),
                                       replace=False))
    print("Plotting:", ", ".join(chosen))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for st in chosen:
        plot_temperature_all(df, date_str, st, save_dir,
                             show=not args.no_show)

if __name__ == "__main__":
    main()
