# Preprocessing and MapReduce

These scripts takes data from CSV files to preprocessed, normalized individual samples.

## Step 0: Get data
Put the CSV files for all the days you want in a folder, we denote this path `FILES_DIR`.

## Step 1: CSV to Pickle files
The first step is to read each csv file line by line to build a by-day dictionary of features grouped by MMSI.

Run:
```bash
python -m src.preprocessing.csv2pkl --input_dir FILES_DIR --output_dir PICKLE_DIR
```

## Step 2:
Preprocess the data using the MapReduce algorithm.

This will map all messages belonging to a MMSI to its own folder, and then concatenate and preprocess (reduction part) and split into individual samples

Run: 
```bash
python -m src.preprocessing.map_reduce --input_dir PICKLE_DIR --temp_dir TEMP_DIR --final_dir final_dir
```

where `TEMP_DIR` and `FINAL_DIR` are respectively a temporary mapping dir and where the preprocessed samples will be written to.

A single sample has the format:
```
{
"mmsi": 12345678,
"traj": nd.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0]])
}
```
inspired by [CIA-Oceanix/TrAISformer](https://github.com/CIA-Oceanix/TrAISformer/).

Columns: LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI

## Step 3: Train test split

Split the preprocessed data into training, validation, and test sets. Using MMSI-aware splitting to prevent data leakage by keeping all tracks from the same vessel in the same partition.

Run:
```bash
python -m src.preprocessing.train_test_split --data_dir FINAL_DIR --val_size 0.1 --test_size 0.1 --random_state 42
```

This creates three subdirectories (`train/`, `val/`, and `test/`) within `FINAL_DIR`.

