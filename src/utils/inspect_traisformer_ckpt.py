import torch
import sys

ckpt_path = "data/checkpoints/traisformer_new_small/traj_traisformer.pt"
try:
    sd = torch.load(ckpt_path, map_location="cpu")
    print("Keys in checkpoint:", sd.keys())
    
    if "config" in sd:
        print("\nConfig in checkpoint:")
        print(sd["config"])
        
    if "state_dict" in sd:
        model_sd = sd["state_dict"]
        print("\nModel state_dict shapes:")
        for k in ["lat_emb.weight", "lon_emb.weight", "sog_emb.weight", "cog_emb.weight", "in_proj.weight"]:
            if k in model_sd:
                print(f"{k}: {model_sd[k].shape}")
            elif "model."+k in model_sd:
                print(f"model.{k}: {model_sd['model.'+k].shape}")
            else:
                print(f"{k} NOT FOUND")
                
except Exception as e:
    print(f"Error loading checkpoint: {e}")
