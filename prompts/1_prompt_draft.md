Here is the steps i have taken and the files i run. 

## CSV to pickle
```bash
python -m src.preprocessing.csv2pkl --input_dir data/raw/ --output_dir data/processed_pickle/
```


## Map_reduce
```bash
python -m src.preprocessing.map_reduce --input_dir  data/processed_pickle/ --temp_dir data/TEMP_DIR --final_dir data/map_reduced/
```


## Train test split
```bash
python -m src.preprocessing.train_test_split --data_dir data/map_reduced/ --val_size 0.1 --test_size 0.1 --random_state 42
```


## Train Model
```bash
python -m src.train.train_traj --config configs/test_alex.yaml
```

# Traj eval
```bash
python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 1 --lon_idx 0 \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures
```


The trajectory evaluation is not working properly. I want a plot of europe, where the actual long and lat are shown together with the predicted long and lat, shown on the map. 

please help me, make this work. 


Read the code and understand the code and the way it workes. 

the data has the columns: LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI



make sure the src.eval.eval_traj_newnewnew script works as it should with the correct figures (map). let me know if you need more information. 


