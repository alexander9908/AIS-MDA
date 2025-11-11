# src/eval/eval_traj_V5.py
from __future__ import annotations
import argparse, os, glob, pickle, csv, datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------- Models ----------------
# Adjust imports to your tree layout
from ..models import TPTrans  # and/or GRU if you keep it
from ..models.traisformer1 import TrAISformer, BinSpec

# ---------------- Water mask (background only) ----------------
# requires: pip install roaring-landmask
from .build_water_mask_V2 import make_water_mask

# ---------------- Style ----------------
plt.rcParams.update({
    "figure.figsize": (7.5, 6.0),
    "axes.edgecolor": "#2a2a2a",
    "axes.labelcolor": "#2a2a2a",
    "xtick.color": "#2a2a2a",
    "ytick.color": "#2a2a2a",
    "font.size": 11,
})

# ---------------- Helpers ----------------
def parse_trip(fname: str) -> Tuple[int, int]:
    base = os.path.basename(fname).replace("_processed.pkl", "")
    mmsi_str, trip_id_str = base.split("_", 1)
    return int(mmsi_str), int(trip_id_str)

def to_iso(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime(fmt)

def load_trip(path: str, min_points: int = 30) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    trip = data["traj"] if isinstance(data, dict) and "traj" in data else np.asarray(data)
    trip = np.asarray(trip)
    if len(trip) < int(min_points):
        raise ValueError(f"too short: {len(trip)} points")
    ts = trip[:, 7]
    if not np.all(ts[:-1] <= ts[1:]):
        trip = trip[np.argsort(ts)]
    return trip

def split_by_percent(trip: np.ndarray, pct: float) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(trip)
    cut = max(1, min(n - 2, int(round(n * pct / 100.0))))
    return trip[:cut], trip[cut:], cut

def robust_extent(lats: np.ndarray, lons: np.ndarray, pad: float = 0.4) -> Tuple[float,float,float,float]:
    lats = lats[np.isfinite(lats)]; lons = lons[np.isfinite(lons)]
    if lats.size == 0 or lons.size == 0:
        return (10.0, 13.0, 55.0, 58.0)
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    if abs(lat_max - lat_min) < 0.2: lat_min -= 0.5; lat_max += 0.5
    if abs(lon_max - lon_min) < 0.2: lon_min -= 0.5; lon_max += 0.5
    lat_min -= pad; lat_max += pad; lon_min -= pad; lon_max += pad
    return (lon_min, lon_max, lat_min, lat_max)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1 = np.radians([lat1, lon1]); p2 = np.radians([lat2, lon2])
    dlat = p2[0]-p1[0]; dlon = p2[1]-p1[1]
    a = np.sin(dlat/2.0)**2 + np.cos(p1[0])*np.cos(p2[0])*np.sin(dlon/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def cumdist(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if len(lat) == 0: return np.array([0.0])
    cd = [0.0]
    for i in range(1, len(lat)):
        cd.append(cd[-1] + haversine_km(lat[i-1], lon[i-1], lat[i], lon[i]))
    return np.asarray(cd, float)

def maybe_denorm_latlon(lat: np.ndarray, lon: np.ndarray,
                        lat_min: Optional[float], lat_max: Optional[float],
                        lon_min: Optional[float], lon_max: Optional[float]) -> Tuple[np.ndarray,np.ndarray]:
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    looks_norm = (np.nanmin(lat) >= -0.1 and np.nanmax(lat) <= 1.1 and
                  np.nanmin(lon) >= -0.1 and np.nanmax(lon) <= 1.1)
    if looks_norm:
        if None in (lat_min, lat_max, lon_min, lon_max):
            raise ValueError("Inputs look normalized — supply --lat_min/--lat_max/--lon_min/--lon_max.")
        lat_deg = lat*(lat_max-lat_min) + lat_min
        lon_deg = lon*(lon_max-lon_min) + lon_min
        return lat_deg, lon_deg
    return lat, lon

def _to_idx_1xT(x: torch.Tensor, device) -> torch.Tensor:
    x = x.to(device).squeeze()
    if x.dim() == 0: x = x.view(1, 1)
    elif x.dim() == 1: x = x.unsqueeze(0)
    elif x.dim() > 2:
        x = x.squeeze()
        if x.dim() == 1: x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(f"Expected <=2 dims for bin indices, got {tuple(x.shape)}")
    return x.to(dtype=torch.long).contiguous()

# ---------------- Model factory ----------------
def build_model(kind: str, ckpt: str, feat_dim: int, horizon: int):
    if kind.lower() == "tptrans":
        model = TPTrans(feat_dim=feat_dim, d_model=192, nhead=4, enc_layers=4, dec_layers=2, horizon=horizon)
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd))
        return model
    if kind.lower() == "traisformer":
        # checkpoint is expected to contain bins spec + model
        sd = torch.load(ckpt, map_location="cpu")
        bins = sd["bins"] if "bins" in sd else sd.get("state_dict", {}).get("bins")
        if not isinstance(bins, BinSpec):
            # try reconstruct
            bins = BinSpec(**sd["bins_dict"])
        model = TrAISformer(
            bins=bins,
            d_model=sd.get("d_model", 512),
            nhead=sd.get("nhead", 8),
            num_layers=sd.get("num_layers", 8),
            dropout=sd.get("dropout", 0.1),
            coarse_merge=sd.get("coarse_merge", 3),
            coarse_beta=sd.get("coarse_beta", 0.2),
        )
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        return model
    raise ValueError(f"unknown model kind: {kind}")

# ---------------- Core per-trip evaluation ----------------
def evaluate_and_plot_trip(
    fpath: str,
    trip: np.ndarray,
    model,
    args,
    sample_idx: int,
) -> Dict[str, Any]:

    mmsi, tid = parse_trip(fpath)

    # ---- split exactly at pred_cut ----
    past, future_true_all, cut = split_by_percent(trip, args.pred_cut)
    if len(past) < 2 or len(future_true_all) < 2:
        raise ValueError("too short after split")

    # Eval tail length (all remaining by default or cap)
    N_future = len(future_true_all) if args.cap_future is None else min(len(future_true_all), int(args.cap_future))

    # Degrees for plotting
    full_lat_deg, full_lon_deg = maybe_denorm_latlon(trip[:,0], trip[:,1],
                                                     args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    lats_past = full_lat_deg[:cut]; lons_past = full_lon_deg[:cut]
    cur_lat = float(lats_past[-1]); cur_lon = float(lons_past[-1])
    lats_true_eval = full_lat_deg[cut:cut+N_future]
    lons_true_eval = full_lon_deg[cut:cut+N_future]
    lats_true_all = full_lat_deg[cut:]; lons_true_all = full_lon_deg[cut:]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device).eval()

    # ---- Predict tail (no fallbacks) ----
    if args.model.lower() == "traisformer":
        seq_in = past[:, :4].astype(np.float32)  # [lat,lon,sog,cog]
        # Convert to degrees first (if normalized)
        lat_deg, lon_deg = maybe_denorm_latlon(
            seq_in[:,0], seq_in[:,1],
            args.lat_min, args.lat_max, args.lon_min, args.lon_max
        )
        # indices
        lat_idx = model.bins.lat_to_bin(torch.tensor(lat_deg, device=device))
        lon_idx = model.bins.lon_to_bin(torch.tensor(lon_deg, device=device))
        # SOG/COG in physical units
        raw_sog, raw_cog = seq_in[:,2], seq_in[:,3]
        sog = (np.clip(raw_sog, 0.0, 1.0) * float(model.bins.sog_max)) if np.nanmax(raw_sog) <= 1.2 else np.clip(raw_sog, 0.0, float(model.bins.sog_max))
        cog = (raw_cog % 1.0) * 360.0 if np.nanmax(np.abs(raw_cog)) <= 1.5 else (raw_cog % 360.0)
        sog_idx = model.bins.sog_to_bin(torch.tensor(sog, device=device))
        cog_idx = model.bins.cog_to_bin(torch.tensor(cog, device=device))

        past_idxs = {
            "lat": _to_idx_1xT(lat_idx, device),
            "lon": _to_idx_1xT(lon_idx, device),
            "sog": _to_idx_1xT(sog_idx, device),
            "cog": _to_idx_1xT(cog_idx, device),
        }

        # sample K candidates, keep best ADE on the eval tail
        best = None
    
        K = max(1, int(getattr(args, "samples", 1)))
        with torch.no_grad():
            for _ in range(K):
                out_idx = model.generate(
                    past_idxs,
                    L=N_future,
                    sampling="sample" if args.temperature > 0 else "greedy",
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                )
                cont = model.bins_to_continuous(out_idx)
                pred_lat = cont["lat"].squeeze(0).cpu().numpy()
                pred_lon = cont["lon"].squeeze(0).cpu().numpy()

                # anchor first predicted point to cut point
                if len(pred_lat) > 0:
                    dlat0 = cur_lat - float(pred_lat[0]); dlon0 = cur_lon - float(pred_lon[0])
                    pred_lat = pred_lat + dlat0; pred_lon = pred_lon + dlon0

                ade_tmp = np.mean([haversine_km(lats_true_eval[i], lons_true_eval[i],
                                                pred_lat[i], pred_lon[i])
                                   for i in range(min(len(pred_lat), len(lats_true_eval)))])
                if (best is None) or (ade_tmp < best[0]):
                    best = (ade_tmp, pred_lat, pred_lon)
        pred_lat, pred_lon = np.asarray(best[1]), np.asarray(best[2])

    elif args.model.lower() == "tptrans":
        # TPTrans outputs normalized absolute lat/lon for H steps; roll until tail finished
        seq_in = past[:, :4].astype(np.float32)  # [lat,lon,sog,cog]

        def looks_norm(x): return (np.nanmin(x) >= -0.05 and np.nanmax(x) <= 1.2)
        seq_norm = seq_in.copy()
        if not (looks_norm(seq_in[:,0]) and looks_norm(seq_in[:,1])):
            if None in (args.lat_min, args.lat_max, args.lon_min, args.lon_max):
                raise ValueError("TPTrans requires normalization bounds; supply --lat_min/--lat_max/--lon_min/--lon_max.")
            seq_norm[:,0] = (seq_in[:,0] - args.lat_min) / float(args.lat_max - args.lat_min)
            seq_norm[:,1] = (seq_in[:,1] - args.lon_min) / float(args.lon_max - args.lon_min)
            # speed/heading normalization (keep if present; otherwise zeros)
            speed_max = float(getattr(args, "speed_max", 30.0))
            seq_norm[:,2] = np.clip(seq_in[:,2] / speed_max, 0.0, 1.0)
            seq_norm[:,3] = (seq_in[:,3] % 360.0) / 360.0

        remaining = int(N_future)
        pred_lat_list, pred_lon_list = [], []

        with torch.no_grad():
            while remaining > 0:
                Tin = min(args.past_len, len(seq_norm))
                X_in = seq_norm[-Tin:, :][None, ...]  # [1,T,4]
                yraw = model(torch.from_numpy(X_in).to(device))[0].cpu().numpy()  # [H,2] normalized absolute
                keep = min(yraw.shape[0], remaining)
                lat_n = np.clip(yraw[:keep, 0], 0.0, 1.0)
                lon_n = np.clip(yraw[:keep, 1], 0.0, 1.0)
                lat_deg, lon_deg = maybe_denorm_latlon(lat_n, lon_n, args.lat_min, args.lat_max, args.lon_min, args.lon_max)

                # anchor first predicted point to cut point
                if len(pred_lat_list) == 0 and keep > 0:
                    dlat0 = cur_lat - float(lat_deg[0]); dlon0 = cur_lon - float(lon_deg[0])
                    lat_deg = lat_deg + dlat0; lon_deg = lon_deg + dlon0

                pred_lat_list.extend(lat_deg.tolist()); pred_lon_list.extend(lon_deg.tolist())

                # feedback loop (teacher-forced on our own predictions)
                lat_n2 = (lat_deg - args.lat_min) / float(args.lat_max - args.lat_min)
                lon_n2 = (lon_deg - args.lon_min) / float(args.lon_max - args.lon_min)
                last_sog = seq_norm[-1,2] if seq_norm.shape[1] > 2 else 0.0
                last_cog = seq_norm[-1,3] if seq_norm.shape[1] > 3 else 0.0
                add_feats = np.stack([lat_n2, lon_n2,
                                      np.full_like(lat_n2, last_sog, dtype=np.float32),
                                      np.full_like(lon_n2, last_cog, dtype=np.float32)], axis=1).astype(np.float32)
                seq_norm = np.vstack([seq_norm, add_feats])
                remaining -= keep

        pred_lat = np.asarray(pred_lat_list, float)
        pred_lon = np.asarray(pred_lon_list, float)

    else:
        raise ValueError("args.model must be 'tptrans' or 'traisformer'.")

    # Optionally trim predicted length to match true evaluation distance
    if args.match_distance and len(pred_lat) > 1 and len(lats_true_eval) > 1:
        dt_true = cumdist(lats_true_eval, lons_true_eval)[-1]
        cd = cumdist(pred_lat, pred_lon)
        # keep as many steps as are within the true distance
        keep = int(np.searchsorted(cd, dt_true, side="right"))
        keep = max(1, min(keep, len(pred_lat)))
        pred_lat, pred_lon = pred_lat[:keep], pred_lon[:keep]

    # ---- Metrics ----
    n_comp = min(len(pred_lat), len(lats_true_eval))
    ade = float(np.mean([haversine_km(lats_true_eval[i], lons_true_eval[i],
                                      pred_lat[i], pred_lon[i]) for i in range(n_comp)])) if n_comp > 0 else np.nan
    fde = float(haversine_km(lats_true_eval[n_comp-1], lons_true_eval[n_comp-1],
                             pred_lat[n_comp-1], pred_lon[n_comp-1])) if n_comp > 0 else np.nan

    # ---- Plot (only true vs predicted lat/lon) ----
    outdir_mmsi = Path(args.out_dir) / f"{mmsi}"
    outdir_mmsi.mkdir(parents=True, exist_ok=True)
    fname_png = outdir_mmsi / f"traj_{args.model}_mmsi-{mmsi}_trip-{tid}_cut-{args.pred_cut}_idx-{sample_idx}.png"

    # extent from past+true+pred
    ext = robust_extent(
        np.concatenate([lats_past, lats_true_all, pred_lat]),
        np.concatenate([lons_past, lons_true_all, pred_lon]),
        pad=0.35 if args.auto_extent else 0.2
    )
    lon_min, lon_max, lat_min, lat_max = ext

    fig, ax = plt.subplots()
    # faint land/water background (non-blocking; if mask fails, we still plot lines)
    try:
        wm = make_water_mask(lat_min, lat_max, lon_min, lon_max, n_lat=256, n_lon=512)
        lat_edges = np.linspace(lat_min, lat_max, 256+1)
        lon_edges = np.linspace(lon_min, lon_max, 512+1)
        # plot land as very light grey
        ax.pcolormesh(lon_edges, lat_edges, (~wm).astype(float),
                      shading="auto", cmap="Greys", alpha=0.12)
    except Exception:
        pass

    ax.plot(lons_past, lats_past, lw=1.6, color="#2a77ff", alpha=0.9, label="past")
    ax.plot(lons_true_all, lats_true_all, lw=2.2, color="#2aaa2a", alpha=0.95, label="true (future)")
    if len(pred_lon) >= 2:
        ax.plot(pred_lon, pred_lat, lw=2.2, color="#d33", alpha=0.95, label=f"pred ({args.model})")
    else:
        ax.scatter(pred_lon, pred_lat, s=20, color="#d33", label=f"pred ({args.model})")

    # cut marker and start of pred
    ax.scatter([lons_past[-1]], [lats_past[-1]], s=36, color="#d33", edgecolor="k", zorder=5)
    ax.scatter([pred_lon[0] if len(pred_lon) else lons_past[-1]],
               [pred_lat[0] if len(pred_lat) else lats_past[-1]],
               s=22, color="#d33", zorder=6, alpha=0.9)

    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_title(f"MMSI {mmsi} · Trip {tid} · Cut {args.pred_cut:.1f}% · ADE {ade:.2f} km · FDE {fde:.2f} km")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(fname_png, dpi=180)
    plt.close(fig)

    return {
        "mmsi": mmsi, "trip": tid, "cut_idx": cut,
        "ade_km": ade, "fde_km": fde,
        "n_future_true": len(lats_true_eval),
        "n_future_pred": len(pred_lat),
        "png": str(fname_png),
    }

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True, help="folder with *_processed.pkl")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", choices=["tptrans","traisformer"], required=True)
    p.add_argument("--horizon", type=int, default=12, help="decoder horizon for tptrans")
    p.add_argument("--past_len", type=int, default=64)
    p.add_argument("--pred_cut", type=float, default=80.0, help="percent split: past/future")
    p.add_argument("--mmsi", type=str, default="", help="comma sep MMSIs to include (optional)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--auto_extent", action="store_true")
    p.add_argument("--match_distance", action="store_true")
    p.add_argument("--samples", type=int, default=1, help="TrAISformer: #samples keep best ADE")
    p.add_argument("--temperature", type=float, default=1.0, help="TrAISformer sampling temperature")
    p.add_argument("--top_k", type=int, default=60, help="TrAISformer top-k sampling")
    p.add_argument("--cap_future", type=int, default=None, help="cap evaluation tail length")
    p.add_argument("--lat_min", type=float, default=None)
    p.add_argument("--lat_max", type=float, default=None)
    p.add_argument("--lon_min", type=float, default=None)
    p.add_argument("--lon_max", type=float, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--speed_max", type=float, default=30.0, help="for TPTrans normalization if speeds present")

    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Collect trips
    files = sorted(glob.glob(os.path.join(args.split_dir, "*_processed.pkl")))
    if args.mmsi.strip():
        allow = set(int(x) for x in args.mmsi.split(","))
        files = [f for f in files if parse_trip(f)[0] in allow]

    # Build model
    feat_dim = 4
    model = build_model(args.model, args.ckpt, feat_dim=feat_dim, horizon=args.horizon)

    # Loop trips
    metrics: List[Dict[str,Any]] = []
    plotted = 0; skipped = 0
    for idx, f in enumerate(files):
        try:
            trip = load_trip(f)
            res = evaluate_and_plot_trip(f, trip, model, args, sample_idx=idx)
            metrics.append(res)
            plotted += 1
            print(f"[ok] saved {res['png']}")
        except Exception as e:
            skipped += 1
            print(f"[skip] {os.path.basename(f)}: {e}")

    # Write summaries
    if metrics:
        sum_trips = Path(args.out_dir) / "summary_trips.csv"
        with open(sum_trips, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(metrics[0].keys()))
            w.writeheader(); w.writerows(metrics)

        # by MMSI
        by = {}
        for m in metrics:
            key = m["mmsi"]
            by.setdefault(key, []).append(m["ade_km"])
        rows = [{"mmsi": k, "n_trips": len(v), "ade_km_mean": float(np.mean(v)), "ade_km_median": float(np.median(v))}
                for k, v in by.items()]
        sum_mmsi = Path(args.out_dir) / "summary_by_mmsi.csv"
        with open(sum_mmsi, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=["mmsi","n_trips","ade_km_mean","ade_km_median"])
            w.writeheader(); w.writerows(rows)

        # overall
        all_ades = [m["ade_km"] for m in metrics]
        overall = {
            "n_trips": len(metrics),
            "ade_km_mean": float(np.mean(all_ades)),
            "ade_km_median": float(np.median(all_ades)),
            "fde_km_mean": float(np.mean([m["fde_km"] for m in metrics])),
        }
        sum_overall = Path(args.out_dir) / "summary_overall.csv"
        with open(sum_overall, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(overall.keys()))
            w.writeheader(); w.writerow(overall)
        print(f"[summary] plotted={plotted} skipped={skipped} total_selected={len(files)}")
        print(f"[summary files] {sum_trips} | {sum_mmsi} | {sum_overall}")
    else:
        print(f"[summary] plotted=0 skipped={skipped} total_selected={len(files)}")

if __name__ == "__main__":
    main()
