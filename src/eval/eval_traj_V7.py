# src/eval/eval_traj_V7.py
# (Use your imports from the original file, I am only showing the changed evaluation logic function)

# ... (Keep Imports and Helper functions like parse_trip, load_trip, etc.) ...

# ---------------- Core per-trip evaluation ----------------
def evaluate_and_plot_trip(
    fpath: str,
    trip: np.ndarray,
    model,
    args,
    sample_idx: int,
) -> Dict[str, Any]:

    mmsi, tid = parse_trip(fpath)
    past, future_true_all, cut = split_by_percent(trip, args.pred_cut)
    
    # ... (Keep existing normalization/LatLon setup logic) ...
    # Assume we have: lats_past, lons_past, cur_lat, cur_lon set up correctly
    
    # Setup plotting coordinates
    full_lat_deg = trip[:,0] # Assuming raw trip is deg for simplicity, adjust if normalized
    full_lon_deg = trip[:,1]
    # (If your data is normalized, ensure you de-normalize 'past' before this point)
    # ... 

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device).eval()

    # ---- PREDICT WITH TPTRANS (DELTA MODE) ----
    if args.model.lower() == "tptrans":
        seq_in = past[:, :4].astype(np.float32)
        
        # Normalize Input (Manual Norm logic from your original script)
        if getattr(args, "lat_min", None) is not None:
             seq_norm = seq_in.copy()
             seq_norm[:,0] = (seq_in[:,0] - args.lat_min) / float(args.lat_max - args.lat_min)
             seq_norm[:,1] = (seq_in[:,1] - args.lon_min) / float(args.lon_max - args.lon_min)
             # SOG/COG Norm...
             speed_max = 30.0
             seq_norm[:,2] = np.clip(seq_in[:,2] / speed_max, 0.0, 1.0)
             seq_norm[:,3] = (seq_in[:,3] % 360.0) / 360.0
        else:
             seq_norm = seq_in

        # Prepare Tensor
        Tin = min(args.past_len, len(seq_norm))
        X_in = seq_norm[-Tin:, :][None, ...] # [1, T, 4]
        X_tensor = torch.from_numpy(X_in).to(device)

        # PREDICT DELTAS
        with torch.no_grad():
            # Output is [1, Horizon, 2] (DeltaLat_Norm, DeltaLon_Norm)
            deltas_pred = model(X_tensor)[0].cpu().numpy() 
            deltas_pred = deltas_pred / 100.0

        # ---- INTEGRATE AND PROJECT ----
        pred_lat_list = []
        pred_lon_list = []
        
        # We must denormalize the DELTAS into Degrees
        # Delta_Deg = Delta_Norm * (Max - Min)
        lat_span = args.lat_max - args.lat_min
        lon_span = args.lon_max - args.lon_min
        
        deltas_deg = deltas_pred.copy()
        deltas_deg[:, 0] *= lat_span
        deltas_deg[:, 1] *= lon_span

        prev_lat, prev_lon = cur_lat, cur_lon

        for k in range(len(deltas_deg)):
            # 1. Apply Model Prediction (Integration)
            dlat = deltas_deg[k, 0]
            dlon = deltas_deg[k, 1]
            
            cand_lat = prev_lat + dlat
            cand_lon = prev_lon + dlon

            # 2. Water Projection (Standard)
            # (Note: I REMOVED the "MIN_STEP_KM" fake movement block completely)
            if is_water(cand_lat, cand_lon):
                fix_lat, fix_lon = cand_lat, cand_lon
            else:
                fix_lat, fix_lon = project_to_water(prev_lat, prev_lon, cand_lat, cand_lon)

            pred_lat_list.append(fix_lat)
            pred_lon_list.append(fix_lon)
            
            # Update prev for next step autoregression
            prev_lat, prev_lon = fix_lat, fix_lon

        pred_lat = np.array(pred_lat_list)
        pred_lon = np.array(pred_lon_list)

    # ... (Rest of the plotting logic remains the same) ...
    # ...