

when i run:
python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 0 --lon_idx 1 \
  --y_order latlon \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures \
  --auto_extent --extent_source actual --extent_outlier_sigma 3.0

it zooms correctly but the prediction line is nowhare to be seen, im not shure if its out of the frame or not there. it is important that the prediction continues from the blue line. 


and when i run:
python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 0 --lon_idx 1 \
  --y_order latlon \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures

i see the eu map correctly but the prediction line is nowhere to be seen, again im not shure if it is because of the pred being by equator (out of frame) or not there.


please help me so the predicted line continues from the last blue dot. 


it is important that the prediction gets placed at the end of the true past and scaled correctly. 