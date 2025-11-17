"""
Builds a water mask for the Denmark region using roaring-landmask.

This script creates a simple raster image representing land and water,
which can be used as a fallback background for trajectory visualizations.
It is a self-contained solution that does not require external shapefiles.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from roaring_landmask import RoaringLandmask

# Define the geographic bounds for the Denmark region
LAT_MIN, LAT_MAX = 54.0, 59.0
LON_MIN, LON_MAX = 5.0, 17.0

# Define image resolution
IMG_WIDTH = 800
IMG_HEIGHT = int(IMG_WIDTH * (LAT_MAX - LAT_MIN) / (LON_MAX - LON_MIN) * 0.6)

def create_water_mask(output_path: str | Path):
    """
    Generates and saves a water mask image using roaring-landmask.

    Args:
        output_path: Path to save the output PNG image.
    """
    print("Generating water mask with roaring-landmask...")
    
    # Create a grid of coordinates
    lons = np.linspace(LON_MIN, LON_MAX, IMG_WIDTH)
    lats = np.linspace(LAT_MIN, LAT_MAX, IMG_HEIGHT)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Flatten the grid for the landmask query
    points = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
    
    # Use roaring-landmask to check which points are on land
    print("Querying landmask for all points...")
    landmask = RoaringLandmask()
    is_land = landmask.contains_many(points[:, 0], points[:, 1])
    
    # Reshape the boolean mask back to the grid shape
    land_mask_grid = is_land.reshape((IMG_HEIGHT, IMG_WIDTH))
    
    # Create a plot to render the mask
    fig, ax = plt.subplots(figsize=(IMG_WIDTH / 100, IMG_HEIGHT / 100), dpi=100)
    
    # Use imshow to display the land mask
    # Water will be 0 (False), Land will be 1 (True)
    # We use a colormap to define land and water colors
    cmap = mcolors.ListedColormap(['#c5e3ff', '#f0f0f0']) # Water, Land
    
    ax.imshow(land_mask_grid, extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), 
              origin='lower', cmap=cmap, interpolation='none')

    ax.set_aspect('equal')
    ax.axis('off')

    # Save the figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving water mask to {output_path}...")
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Water mask created successfully.")

def main():
    parser = argparse.ArgumentParser(description="Generate a water mask for trajectory plotting using roaring-landmask.")
    parser.add_argument(
        "--output",
        default="kalman_filter/assets/water_mask.png",
        help="Path to save the output water mask PNG file."
    )
    args = parser.parse_args()
    
    create_water_mask(args.output)

if __name__ == "__main__":
    main()
