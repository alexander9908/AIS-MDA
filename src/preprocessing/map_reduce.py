import numpy as np
import os
import pickle
import shutil
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from src.preprocessing.preprocessing import process_single_mmsi_track

# Define the columns for our toy data for clarity
# Original: LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI
# Toy:      TIMESTAMP, LAT, LON, SOG
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI  = list(range(9))

def cleanup(force=False):
    """Removes all the directories and files created by the script."""
    print("--- Cleaning Up ---")
    for d in [TEMP_DIR, FINAL_DIR]:
        if os.path.exists(d):
            if force:
                shutil.rmtree(d)
                print(f"Removed {d}")
            else:
                raise Exception(f"Directory {d} already exists. Use --cleanup to remove it.")
    print("-------------------------\n")

def stage_1_map_and_shuffle():
    """
    Goes through all input files and re-sorts them by MMSI
    into a temporary directory.
    
    INPUT:
    - toy_input_data/file1.pkl
    - toy_input_data/file2.pkl
    
    OUTPUT:
    - temp_mmsi_groups/001/file1.pkl
    - temp_mmsi_groups/002/file1.pkl
    - temp_mmsi_groups/002/file2.pkl
    - temp_mmsi_groups/003/file2.pkl
    - ...
    """
    print("--- Stage 1: Map & Shuffle (Grouping by MMSI) ---")
    
    # Clean slate for the temporary shuffle directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    input_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".pkl")]

    for file_path in tqdm(input_files, desc="Processing input files"):
        print(f"  Mapping {file_path}...")
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)
            
            # If IS_TEST sample 1% of MMSIs
            if IS_TEST:
                sampled_mmsis = np.random.choice(list(data_dict.keys()), size=max(1, len(data_dict)//100), replace=False)
                data_dict = {mmsi: data_dict[mmsi] for mmsi in sampled_mmsis}
            for mmsi, track_segment in tqdm(data_dict.items(), desc="  Processing MMSIs", leave=False):
                # Create a directory for this specific MMSI
                mmsi_dir = os.path.join(TEMP_DIR, str(mmsi))
                os.makedirs(mmsi_dir, exist_ok=True)
                
                # Save this segment into the MMSI's folder
                # We name it after the original file to avoid collisions
                segment_filename = os.path.basename(file_path)
                output_path = os.path.join(mmsi_dir, segment_filename)
                
                with open(output_path, "wb") as out_f:
                    pickle.dump(track_segment, out_f)
    if VESSEL_TYPE_DIR is not None:
        vessel_type_files = [f for f in os.listdir(VESSEL_TYPE_DIR) if f.endswith(".pkl")]
        for vt_file in tqdm(vessel_type_files, desc="Copying vessel type files"):
            with open(os.path.join(VESSEL_TYPE_DIR, vt_file), "rb") as f_src:
                vessel_types = pickle.load(f_src)
            for mmsi in vessel_types.keys():
                mmsi_dir = os.path.join(TEMP_DIR, str(mmsi), 'vessel_types')
                os.makedirs(mmsi_dir, exist_ok=True)
                vessel_type_filename = os.path.join(mmsi_dir, vt_file)
                with open(vessel_type_filename, "ab") as f_dst:
                    pickle.dump({mmsi: vessel_types[mmsi]}, f_dst)
                    
    print("  Map & Shuffle Complete.")
    print("-------------------------\n")
    
def process_single_mmsi(mmsi_info):
    """
    Wrapper to unpack arguments for multiprocessing.
    """
    mmsi, mmsi_dir_path, final_dir = mmsi_info
    
    # --- Load all segments for this MMSI ---
    all_segments = []
    segment_files = [f for f in os.listdir(mmsi_dir_path) if f.endswith(".pkl") and not f.startswith("vessel_types_")]
    if not segment_files:
        return None
    
    for seg_file in segment_files:
            segment_path = os.path.join(mmsi_dir_path, seg_file)
            with open(segment_path, "rb") as f:
                track_segment = pickle.load(f)
                all_segments.append(track_segment)
    
    # --- Merge into one giant track ---
    # This is the *only* point where all data for a
    # single MMSI is in memory.
    try:
        full_track = np.concatenate(all_segments, axis=0)
    except ValueError:
        tqdm.write(f"    MMSI {mmsi}: Error concatenating. Skipping.")
        return None

    # --- Run processing ---
    processed_data = process_single_mmsi_track(mmsi, full_track)
    
    # --- Save final result ---
    if processed_data:
        for k, traj in processed_data.items():
            final_output_path = os.path.join(final_dir, f"{mmsi}_{k}_processed.pkl")
            data_item = {'mmsi': mmsi, 'traj': traj}
            with open(final_output_path, "wb") as f:
                pickle.dump(data_item, f)
        return True
    return None
    
def stage_2_reduce(n_workers: int = None):
    """
    Loops through each MMSI folder in the TEMP_DIR *one at a time*.
    Loads all data for *only* that MMSI, runs the processing,
    and saves the final result.
    """
    print("--- Stage 2: Reduce (Processing by MMSI) ---")
    
    os.makedirs(FINAL_DIR, exist_ok=True)
    
    mmsi_folders = os.listdir(TEMP_DIR)
    
    # Prepare list of (mmsi, path, output_dir) tuples for parallel processing
    mmsi_tasks = []
    for mmsi in mmsi_folders:
        mmsi_dir_path = os.path.join(TEMP_DIR, mmsi)
        if os.path.isdir(mmsi_dir_path):
            mmsi_tasks.append((mmsi, mmsi_dir_path, FINAL_DIR))
            
    # Determine number of workers
    if n_workers is None:
        n_workers = cpu_count()
    print(f"  Using {n_workers} parallel workers")
    
    # Process in parallel
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_mmsi, mmsi_tasks),
            total=len(mmsi_tasks),
            desc="Processing MMSIs"
        ))
    
    print("  Reduce Complete.")
    print("-------------------------\n")
    

def combine_vessel_types():
    """
    Combines all individual vessel type files into a single dictionary.
    """
    vessel_types = dict()
    vessel_types_files = [f for f in os.listdir(VESSEL_TYPE_DIR)] if VESSEL_TYPE_DIR is not None else []
    for vt_file in tqdm(vessel_types_files, desc="Combining vessel type files"):
        vt_path = os.path.join(VESSEL_TYPE_DIR, vt_file)
        with open(vt_path, "rb") as f:
            vt_data = pickle.load(f)
            vessel_types.update(vt_data)
    combined_path = os.path.join(FINAL_DIR, "vessel_types.pkl")
    with open(combined_path, "wb") as f:
        pickle.dump(vessel_types, f)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Map-Reduce preprocessing for AIS data.")
    parser.add_argument("--input_dir", type=str, default="data/pickle_files", help="Directory containing input pickle files.")
    parser.add_argument("--temp_dir", type=str, default="data/map_reduce_temp", help="Temporary directory for grouped MMSI data.")
    parser.add_argument("--final_dir", type=str, default="data/map_reduce_final", help="Final output directory for processed MMSI tracks.")
    parser.add_argument("--cleanup", action="store_true", help="Clean up temporary files and directories after processing.")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only a subset of MMSIs).")
    args = parser.parse_args()
    
    IS_TEST = args.test
    INPUT_DIR = args.input_dir
    TEMP_DIR = args.temp_dir
    FINAL_DIR = args.final_dir
    VESSEL_TYPE_DIR = os.path.join(INPUT_DIR, 'vessel_types') if os.path.exists(os.path.join(INPUT_DIR, 'vessel_types')) else None

    cleanup(force = args.cleanup)
    stage_1_map_and_shuffle()
    stage_2_reduce()
    if VESSEL_TYPE_DIR is not None:
        combine_vessel_types()
    
    # Remove temporary directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Removed temporary directory {TEMP_DIR}")