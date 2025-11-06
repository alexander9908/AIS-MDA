
The plots are not correct, when i display Europe:

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 1 --lon_idx 0 \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures

i see a map of eu but no trajectories, nothing. 

then when i choose to zoom in:

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 1 --lon_idx 0 \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures \
  --auto_extent  # omit to see the full Europe view (-25, 45, 30, 72)

i see the trajectories but the map is blank, i want to zoom in on that region where the trajectoryt is, fx zoom in on denmark etc. 


also i think the predicted values are normalised instead of correct long and lat. 


please help me make it work as it should. 


in src/preprocessing/preprocessing.py we have a de_normalize_track function we can use. 

def de_normalize_track(track: np.ndarray) -> np.ndarray:
    """Denormalizes a single track."""
    denorm_track = copy.deepcopy(track)

    denorm_track[:, LAT] = denorm_track[:, LAT] * (LAT_MAX - LAT_MIN) + LAT_MIN
    denorm_track[:, LON] = denorm_track[:, LON] * (LON_MAX - LON_MIN) + LON_MIN
    denorm_track[:, SOG] = denorm_track[:, SOG] * SPEED_MAX
    denorm_track[:, COG] = denorm_track[:, COG] * 360.0
    
    return denorm_track


review all the code and make shure it works properly. 