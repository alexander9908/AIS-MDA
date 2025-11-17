"""
Builds a water mask for Denmark region for map-based plotting.

This script uses shapefiles to create a simple raster image representing land and water,
which can be used as a fallback background for trajectory visualizations when a live
basemap service (like contextily) is unavailable.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box

# Define the geographic bounds for the Denmark region
LAT_MIN, LAT_MAX = 54.0, 59.0
LON_MIN, LON_MAX = 5.0, 17.0

# Define image resolution
IMG_WIDTH = 800
IMG_HEIGHT = int(IMG_WIDTH * (LAT_MAX - LAT_MIN) / (LON_MAX - LON_MIN) * 0.6)

def create_water_mask(shapefile_path: str, output_path: str):
    """
    Generates and saves a water mask image from a coastline shapefile.

    Args:
        shapefile_path: Path to the world coastline shapefile.
        output_path: Path to save the output PNG image.
    """
    print(f"Loading shapefile from {shapefile_path}...")
    try:
        world = gpd.read_file(shapefile_path)
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        print("Please ensure you have a valid shapefile. You can download one from sites like Natural Earth.")
        return

    # Define the bounding box for our area of interest
    denmark_bbox = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

    # Filter geometries to those intersecting the bounding box
    print("Clipping geometries to the Denmark region...")
    land_polygons = world[world.intersects(denmark_bbox)].clip(denmark_bbox)

    if land_polygons.empty:
        print("No land polygons found in the specified bounding box.")
        return

    # Create a plot to render the mask
    fig, ax = plt.subplots(figsize=(IMG_WIDTH / 100, IMG_HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor('#c5e3ff')  # Water color
    ax.set_facecolor('#c5e3ff')

    # Plot the land polygons
    land_polygons.plot(ax=ax, color='#f0f0f0', edgecolor='gray', linewidth=0.5)

    # Configure the plot to match the geographic bounds
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect('equal')
    ax.axis('off')  # No axes or borders

    # Save the figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving water mask to {output_path}...")
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Water mask created successfully.")

def main():
    parser = argparse.ArgumentParser(description="Generate a water mask for trajectory plotting.")
    parser.add_argument(
        "--shapefile",
        required=True,
        help="Path to the coastline shapefile (e.g., from Natural Earth)."
    )
    parser.add_argument(
        "--output",
        default="kalman_filter/assets/water_mask.png",
        help="Path to save the output water mask PNG file."
    )
    args = parser.parse_args()
    
    create_water_mask(args.shapefile, args.output)

if __name__ == "__main__":
    main()
    print("\nTo use this mask, you will need a shapefile.")
    print("A good option is the Natural Earth 'Land' dataset:")
    print("1. Go to: https://www.naturalearthdata.com/downloads/10m-physical-vectors/")
    print("2. Download 'Land' (ne_10m_land.zip)")
    print("3. Unzip it and provide the path to 'ne_10m_land.shp' to this script.")
    print(f"\nExample: python -m kalman_filter.build_water_mask --shapefile /path/to/ne_10m_land.shp")
