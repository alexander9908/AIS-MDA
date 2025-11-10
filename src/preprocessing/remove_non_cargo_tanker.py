import pickle
import os
import numpy as np

def remove_non_cargo_tanker(preprocessed_dir):
    
    try: # load vessel types
        vessel_types_path = os.path.join(preprocessed_dir, "vessel_types.pkl")
        with open(vessel_types_path, "rb") as f:
            vessel_types = pickle.load(f)
    except FileNotFoundError:
        print(f"Vessel types file not found at {vessel_types_path}.")
        return
    
    if 'train' in (os.listdir(preprocessed_dir) and os.path.isdir(os.path.join(preprocessed_dir, 'train'))) and \
        len(os.listdir(os.path.join(preprocessed_dir, 'train'))) > 0: # Train/val/test split exists
        # Unsplit
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(preprocessed_dir, split)
            for fname in os.listdir(split_dir):
                src_path = os.path.join(split_dir, fname)
                dst_path = os.path.join(preprocessed_dir, fname)
                os.rename(src_path, dst_path)
            os.rmdir(split_dir)
    cargo_codes = set(np.arange(70, 80))
    tanker_codes = set(np.arange(80, 90))
    fishing_codes = set([30])
    
    cargo_count = 0
    tanker_count = 0
    fishing_count = 0
    other_count = 0
    
    removed_vessels_dir = os.path.join(preprocessed_dir, "removed_vessels")
    os.makedirs(removed_vessels_dir, exist_ok=True)
    
    def remove_file(file_name):
        file_path = os.path.join(preprocessed_dir, file_name)
        dst_path = os.path.join(removed_vessels_dir, file_name)
        os.rename(file_path, dst_path)
    
    # Filter cargo tanker
    for fname in os.listdir(preprocessed_dir):
        if not fname.endswith('.npy'):
            continue
        vessel_mmsi = int(fname.split('_')[0])
        vessel_type = vessel_types.get(vessel_mmsi, 0)
        if vessel_type in cargo_codes:
            cargo_count += 1
        elif vessel_type in tanker_codes:
            tanker_count += 1
        elif vessel_type in fishing_codes:
            fishing_count += 1
            remove_file(fname)
        else:
            other_count += 1
            remove_file(fname)
    
    print(f"Cargo voyages kept: {cargo_count}")
    print(f"Tanker voyages kept: {tanker_count}")
    print(f"Fishing voyages removed: {fishing_count}")
    print(f"Other voyages removed: {other_count}")
    print("Finished removing non-cargo/tanker voyages.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove non-cargo/tanker voyages from preprocessed data.")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Path to the preprocessed data directory.")
    args = parser.parse_args()
    
    remove_non_cargo_tanker(args.preprocessed_dir)