import numpy as np
import os
import pickle
import shutil
from tqdm import tqdm
import argparse
from src.preprocessing.preprocessing_V2 import process_single_mmsi_track

# Column indices (match the rest of the project)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI  = list(range(9))

def cleanup(force=False, temp_dir=None, final_dir=None):
    print("--- Cleaning Up ---")
    for d in [temp_dir, final_dir]:
        if not d:
            continue
        if os.path.exists(d):
            if force:
                shutil.rmtree(d)
                print(f"Removed {d}")
            else:
                raise Exception(f"Directory {d} already exists. Use --cleanup to remove it.")
    print("-------------------------\n")

def _list_pickles(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pkl")]

def stage_1_map_and_shuffle(input_dir, temp_dir):
    """
    Same behavior as your original: group each input file's MMSI segments under temp_dir/<mmsi>/
    """
    print("--- Stage 1: Map & Shuffle (Grouping by MMSI) ---")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    input_files = _list_pickles(input_dir)

    for file_path in tqdm(input_files, desc="Processing input files"):
        print(f"  Mapping {file_path}...")
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

            for mmsi, track_segment in tqdm(data_dict.items(), desc="  Processing MMSIs", leave=False):
                mmsi_dir = os.path.join(temp_dir, str(mmsi))
                os.makedirs(mmsi_dir, exist_ok=True)

                segment_filename = os.path.basename(file_path)
                output_path = os.path.join(mmsi_dir, segment_filename)

                with open(output_path, "wb") as out_f:
                    pickle.dump(track_segment, out_f)

    print("  Map & Shuffle Complete.")
    print("-------------------------\n")

def _concat_sort_dedup(segments):
    """
    Concatenate segments, sort by TIMESTAMP, drop exact duplicate rows.
    Returns merged array as np.ndarray.
    """
    full = np.concatenate(segments, axis=0)
    # Sort by time (stable)
    order = np.argsort(full[:, TIMESTAMP].astype(np.float64, copy=False), kind="mergesort")
    full = full[order]
    # Drop exact duplicate rows (rare, but safe)
    if len(full) > 1:
        dup = np.all(full[1:] == full[:-1], axis=1)
        if np.any(dup):
            keep = np.ones(len(full), dtype=bool)
            keep[1:][dup] = False
            full = full[keep]
    return full

def stage_2_reduce(temp_dir, final_dir):
    """
    Same high-level flow as your original script (map_reduce.py) :contentReference[oaicite:3]{index=3}:
      - Iterate MMSI folders
      - Load all segments for that MMSI
      - Concatenate into one track (the only point where the whole MMSI is in memory)
      - Call process_single_mmsi_track(mmsi, full_track)
      - Save results as <mmsi>_<k>_processed.pkl with {'mmsi': mmsi, 'traj': traj}
    """
    print("--- Stage 2: Reduce (Processing by MMSI) ---")
    os.makedirs(final_dir, exist_ok=True)

    mmsi_folders = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]

    for mmsi in tqdm(mmsi_folders, desc="Processing MMSIs"):
        mmsi_dir_path = os.path.join(temp_dir, mmsi)

        # Load all segments for this MMSI
        segments = []
        for seg_file in _list_pickles(mmsi_dir_path):
            with open(seg_file, "rb") as f:
                arr = np.asarray(pickle.load(f))
                # Defensive checks
                if arr.ndim != 2 or arr.shape[1] < 9:
                    tqdm.write(f"    MMSI {mmsi}: Bad segment shape {arr.shape}. Skipping this segment.")
                    continue
                segments.append(arr)

        if not segments:
            tqdm.write(f"    MMSI {mmsi}: No valid segments. Skipping.")
            continue

        # Merge into one track (only point where full MMSI is in memory)
        try:
            full_track = _concat_sort_dedup(segments)
        except Exception as e:
            tqdm.write(f"    MMSI {mmsi}: Concatenation error: {e}. Skipping.")
            continue

        if len(full_track) > 150_000:
            tqdm.write(f"    MMSI {mmsi}: Large track with {len(full_track):,} points. Processing...")

        # Run your existing processing pipeline (unchanged API) :contentReference[oaicite:4]{index=4}
        try:
            processed_data = process_single_mmsi_track(int(mmsi), full_track)
        except Exception as e:
            tqdm.write(f"    MMSI {mmsi}: Processing error: {e}. Skipping.")
            continue

        # Save final result (same format as your current script) :contentReference[oaicite:5]{index=5}
        if processed_data:
            for k, traj in processed_data.items():
                out_path = os.path.join(final_dir, f"{mmsi}_{k}_processed.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump({'mmsi': int(mmsi), 'traj': traj}, f)

    print("  Reduce Complete.")
    print("-------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Map-Reduce preprocessing for AIS data (V2).")
    parser.add_argument("--input_dir", type=str, default="data/pickle_files",
                        help="Directory containing input pickle files.")
    parser.add_argument("--temp_dir", type=str, default="data/map_reduce_temp",
                        help="Temporary directory for grouped MMSI data.")
    parser.add_argument("--final_dir", type=str, default="data/map_reduce_final",
                        help="Final output directory for processed MMSI tracks.")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up temporary files and directories before processing.")
    args = parser.parse_args()

    cleanup(force=args.cleanup, temp_dir=args.temp_dir, final_dir=args.final_dir)
    stage_1_map_and_shuffle(args.input_dir, args.temp_dir)
    stage_2_reduce(args.temp_dir, args.final_dir)

if __name__ == "__main__":
    main()