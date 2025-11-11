


when i run

python -m src.eval.eval_traj_V6 \
  --split_dir data/map_reduced/test \
  --ckpt data/checkpoints/traj_traisformer.pt \
  --model traisformer \
  --pred_cut 80 \
  --samples 24 --temperature 1.05 --top_k 60 \
  --out_dir data/figures/traisformer_v6_test \
  --auto_extent --match_distance --mmsi 209536000,209982000


[skip] 209536000_0_processed.pkl: expected Tensor as element 0 in argument 0, but got float
[skip] 209982000_0_processed.pkl: expected Tensor as element 0 in argument 0, but got float
[summary] plotted=0 skipped=2 total_selected=2



please help me find the error. 



i think it is in the @traisformer1.py or the eval_traj_V6.py. plese help me understand and fix the problem. 



# output

I reproduced your error and traced it to a name collision inside the TrAISformer
  sampler.

  What’s happening

  - In src/models/traisformer1.py: generate(), the decoder input sequence is
    stored in a tensor named y. Later, inside the inner loop where the bearing is
    computed, a temporary float variable is also named y. That float shadows the
    tensor y. When the code tries to append the next token with torch.cat([y, ...],
    dim=1), the first element is now a float, triggering: “expected Tensor as
    element 0 in argument 0, but got float”.

  Fix applied

  - Renamed the decoder sequence tensor y to y_seq.
  - Renamed the temporary bearing variables to y_bear and x_bear to avoid any
    future collisions.
  - This removes the float/tensor collision and makes torch.cat operate on tensors
    as intended.

  Changed lines

  - src/models/traisformer1.py:309
      - y → y_seq for the decoder sequence: y_seq = self.start_token.expand(B,
        1, -1)
  - src/models/traisformer1.py:345–346
      - y → y_seq in the decoder call: self.decoder(tgt=self.posenc(y_seq), ...)
  - src/models/traisformer1.py:383–385
      - y, x → y_bear, x_bear in the bearing computation and atan2 call
  - src/models/traisformer1.py:498
      - y → y_seq in the concatenation: y_seq = torch.cat([y_seq,
        self.in_proj(step_embed)], dim=1)

  Verification

  - I validated the fix by directly loading your checkpoint and running
    model.generate() on one of the test trips. It now returns valid shapes:
      - {'lat': torch.Size([1, 4]), 'lon': torch.Size([1, 4]), 'sog':
        torch.Size([1, 4]), 'cog': torch.Size([1, 4])}
  - Running the full plotting script in this environment sometimes aborts due to
    Matplotlib cache permissions (Abort trap: 6). That’s unrelated to the original
    error. If you hit it locally, set MPLCONFIGDIR to a writable dir, e.g.:
      - MPLCONFIGDIR=data/.mplcache python -m src.eval.eval_traj_V6 ...

  What to run

  - Your original command should work now for TraISformer without the “expected
    Tensor … got float” skips:
      - python -m src.eval.eval_traj_V6 --split_dir data/map_reduced/test
        --ckpt data/checkpoints/traj_traisformer.pt --model traisformer
        --pred_cut 80 --samples 24 --temperature 1.05 --top_k 60 --out_dir
        data/figures/traisformer_v6_test --auto_extent --match_distance --mmsi
        209536000,209982000

  Notes

  - I kept the rest of your eval and model logic intact.
  - If you want, I can add a tiny sampler smoke test to prevent regressions like
    this in the future.




Great — I kept your lines and legend intact and added an optional satellite
  basemap background.

  What to do

  - Install basemap deps (one time):
      - pip install -r env/requirements.txt
  - Run with the satellite style:
      - python -m src.eval.eval_traj_V6 ... --style satellite
  - Other nice options:
      - --style terrain | toner | watercolor | positron | osm
  - Increase figure sharpness if needed:
      - add --dpi 220 (or 300) to your command

  What changed

  - src/eval/eval_traj_V6.py: Adds basemap support via contextily; keeps your
  original solid lines and legend.
  - env/requirements.txt: Adds contextily and xyzservices (only used when style !
  = classic).

  Notes

  - Requires internet to fetch tiles. If offline, use --style classic (fallback
  land/water shading).
  - If Matplotlib complains about cache permissions, set:
      - MPLCONFIGDIR=data/.mplcache python -m src.eval.eval_traj_V6 ... --style
  satellite

  Example

  - python -m src.eval.eval_traj_V6 --split_dir data/map_reduced/test --ckpt data/
  checkpoints/traj_traisformer.pt --model traisformer --pred_cut 80 --samples
  24 --temperature 1.05 --top_k 60 --out_dir data/figures/traisformer_v6_test
  --auto_extent --match_distance --mmsi 209536000,209982000 --style satellite --dpi
  220

  Want me to make satellite the default style instead of classic?






why traisformer so stationary:


• Likely causes

  - Data distribution: Many AIS points are near-stationary (ports/anchorage). Next-
    step CE training makes “stay or move one tiny cell” very likely.
  - Discretization vs cadence: If your cell size is small compared to typical
    5‑min motion, the ground-truth next bin often equals the current (reinforces
    stickiness).
  - Sampler priors: Continuity and turn penalties bias toward minimal motion;
    modest direction push may not overcome this near coasts or with low SOG.
  - Decoding settings: Low temperature and small top_k restrict exploration around
    the current cell.

  What I added (so you can tune without retraining)

  - TrAISformer sampler knobs are now adjustable at runtime. In generate(),
    constants are replaced with parameters, and the eval CLI exposes them.
  - New flags in eval:
      - --lambda_cont: continuity weight (default ~0.04)
      - --alpha_dir: direction push weight (default ~1.60)
      - --beta_turn: turn penalty (default ~0.55)
      - --step_scale: step magnitude scaling (default ~0.70)
      - --sog_floor: knots floor for motion prior (default ~3.0)
      - --no_first_neigh: disables “first-step 8-neighborhood greedy” constraint
      - --allow_same_cell: allows staying in the same cell (not recommended; off
        by default)

  Recommended settings to reduce “shyness”

  - Sampler:
      - --lambda_cont 0.015–0.02 (lower continuity = less stickiness)
      - --alpha_dir 2.0–2.5 (stronger push along heading)
      - --beta_turn 0.35–0.45 (slightly more willing to turn/move)
      - --step_scale 1.0–1.2 (longer step toward heading)
      - --sog_floor 5.0 (avoid tiny steps at low speeds)
      - Consider --no_first_neigh to avoid initial greedy 8-neighbor clamp.
  - Decoding:
      - Increase exploration: --temperature 1.3–1.6, --top_k 100–200 (or omit to
        disable truncation).
      - Keep sampling multiple trajectories: --samples 16–32 and select best ADE.
  - Evaluation sanity:
      - Temporarily drop --match_distance to visualize raw rollout length and
        ensure it’s not being trimmed away.
      - If past is stationary, try a smaller/later --pred_cut or filter trips with
        past SOG > 1 kn to audit behavior.

  Example command

  - More assertive motion, same lines/legend:
      - python -m src.eval.eval_traj_V6 --split_dir data/map_reduced/test
        --ckpt data/checkpoints/traj_traisformer.pt --model traisformer
        --pred_cut 80 --samples 24 --temperature 1.4 --top_k 140 --out_dir data/
        figures/traisformer_v6_tuned --auto_extent --match_distance --mmsi
        209536000,209982000 --style satellite --dpi 220 --lambda_cont 0.02
        --alpha_dir 2.2 --beta_turn 0.40 --step_scale 1.10 --sog_floor 5.0

  Longer-term (training/data)

  - Reduce stationary bias: reweight or sample windows with motion (SOG > 1 kn)
    more often for next-step CE.
  - Consider coarser bins or dynamic binning in areas with very small inter-step
    displacement.
  - Predict deltas (lat/lon) as an auxiliary loss or use scheduled sampling to
    reduce exposure bias.

  If you want, I can make these tuned sampler values the default for your current
  experiments or add a small debug print to log the first few sampled (lat, lon,
  sog) bins to verify movement.





# still stationary 

Thanks for the figure — that confirms the sampler is still hugging the current
  cell and barely stepping out.

  Why it happens

  - Next‑step CE bias: The lat/lon heads are typically peaked around the current
    bin; without a strong counter‑force, the outer-sum grid favors tiny moves.
  - Weak motion prior: If predicted SOG is small (common in training), the heading
    prior’s “aim point” stays close to the current cell.
  - Continuity/turn costs: These regularizers keep trajectories smooth but also
    damp motion unless the direction push is high enough.
  - First‑step clamp: The 8‑neighborhood greedy first step can lock the first move
    to a tiny nudge.

  Quick fixes to try (no retrain)
  Use the new sampler flags to push movement. Start with this “assertive but safe”
  set:

  - Lower stickiness and raise directional push:
      - --lambda_cont 0.015
      - --alpha_dir 2.4
      - --beta_turn 0.40
  - Increase step magnitude and floor the speed:
      - --step_scale 1.15
      - --sog_floor 6.0
  - Relax the first-step clamp and allow broader exploration:
      - --no_first_neigh
      - --temperature 1.4 --top_k 160

  Example command

  - python -m src.eval.eval_traj_V6 \
    --split_dir data/map_reduced/test \
    --ckpt data/checkpoints/traj_traisformer.pt \
    --out_dir data/figures/traisformer_v6_tuned_01 \
    --model traisformer \
    --pred_cut 80 \
    --samples 24 --temperature 1.4 --top_k 160 \
    --lambda_cont 0.015 --alpha_dir 2.4 --beta_turn 0.40 --step_scale 1.15
    --sog_floor 6.0 --no_first_neigh \
    --auto_extent --match_distance \
    --mmsi 209982000 \
    --style satellite --dpi 220 
    

  If it still looks too stationary

  - Push further:
      - --lambda_cont 0.01
      - --alpha_dir 3.0
      - --step_scale 1.25
      - --sog_floor 8.0
      - Optionally increase exploration: --top_k 220; or try greedy once:
        --temperature 0
  - Temporarily drop --match_distance to inspect raw rollout length.

  Sanity checks

  - Past motion: If the cut happens at near‑docking (low past SOG), even higher
    priors may be needed. Try a different trip or a lower pred_cut (e.g., 70) to
    see behavior during steady cruising.
  - Water mask: If the water mask is too conservative in tight channels, it can
    block good exits; try briefly with --allow_same_cell (not recommended for final
    runs) just to confirm it’s not masking.

  Longer-term improvements (optional)

  - Reweight training windows toward in‑motion segments (SOG > 1 kn).
  - Add a small CE term on SOG bins to discourage collapsing to low speeds.
  - Increase bin size or use non‑uniform bins in slow areas to make one‑cell moves
    more meaningful.

  If you want, I can make one of the tuned settings the default in the script, or
  add a debug print that logs per‑step bin hops and the per‑step km distance so we
  can quickly see whether it’s moving enough.