import torch
import sys
import os

checkpoint_path = r"d:\DTU\AIS-MDA\data\checkpoints\hpc_1_delta\traj_tptrans_delta.pt"

print(f"Loading checkpoint from {checkpoint_path}...")

try:
    # Load on CPU to avoid CUDA errors if not available
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print("Keys in checkpoint:", checkpoint.keys())
        
        if 'config' in checkpoint:
            print("\nConfig found:")
            print(checkpoint['config'])
            
        if 'hyper_parameters' in checkpoint:
            print("\nHyperparameters found:")
            print(checkpoint['hyper_parameters'])
            
        if 'scale_factor' in checkpoint:
            print("\nscale_factor found:")
            print(checkpoint['scale_factor'])

        if 'norm_config' in checkpoint:
            print("\nnorm_config found:")
            print(checkpoint['norm_config'])

        # Check for model state dict to see structure
        if 'model_state_dict' in checkpoint:
            print("\nModel state dict keys (first 5):")
            print(list(checkpoint['model_state_dict'].keys())[:5])
        elif 'state_dict' in checkpoint:
            print("\nState dict keys (first 5):")
            print(list(checkpoint['state_dict'].keys())[:5])
            
    else:
        print("Checkpoint is not a dictionary. It might be just the model state dict.")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
