python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --max_plots 8 \
    --out_dir data/figures \
    --auto_extent --extent_outlier_sigma 3.0 \
    --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
    --log_skip_reasons --seed 0
[mode] multi
[mode] multi  found=87  selected=8  max_plots=8
[skip] 209867000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 211404810_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 211770520_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 219017664_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 219024178_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 220464000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 244338000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 257861000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[summary] plotted=0, skipped=8, total=8




python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --mmsi 209867000 --trip_id 0 \
    --pred_cut 90 \
    --auto_extent \
    --lat_idx 0 --lon_idx 1 --y_order latlon \
    --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
    --annotate_id --log_skip_reasons
[mode] single
[mode] single  found=87  selected=1  max_plots=8
[skip] 209867000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[summary] plotted=0, skipped=1, total=1







python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --mmsi all --max_plots 20 \
    --pred_cut 85 --iter_rollout \
    --auto_extent \
    --out_dir data/figures \
    --output_per_mmsi_subdir \
    --save_meta --meta_path data/figures/traj_eval_meta.csv \
    --log_skip_reasons --seed 42
[mode] batch_all
[mode] batch_all  found=87  selected=20  max_plots=20
[skip] 209114000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 209867000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 211363300_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 211404810_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 211436320_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 211440360_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 211770520_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 211815680_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 211911000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 212673000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 215114000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 215253000_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 218481000_1_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 219000431_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 219001518_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 219002827_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 219005671_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 219006113_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 219006971_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[skip] 219008474_0_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value
[summary] plotted=0, skipped=20, total=20