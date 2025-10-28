# For plotting trajectories

import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from datetime import datetime


# --- Constants for data generation ---
# Column indices based on your description
LAT_IDX = 0
LON_IDX = 1
TIMESTAMP_IDX = 7
NUM_COLS = 9

# --- Plot boundaries from user (Updated to focus on Denmark) ---
LAT_MIN = 54.5
LAT_MAX = 58.0
LON_MIN = 8.0
LON_MAX = 13.0

def plot_trajectory(tracks):
    """
    Plots a trajectory from a (n, 9) NumPy array.
    """
    
    # 1. Extract the data
    lats = tracks[:, LAT_IDX]
    lons = tracks[:, LON_IDX]
    timestamps = tracks[:, TIMESTAMP_IDX]
    
    # --- Calculate mean latitude for aspect correction ---
    mean_lat = np.mean(lats)

    print("Data extracted. Plotting...")

    # 2. Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 3. Create a scatter plot, color-coded by time
    # 'c=timestamps' maps the timestamp to a color
    # 'cmap='viridis'' is the color map (yellow for high, blue for low)
    scatter = ax.scatter(lons, lats, c=timestamps, cmap='viridis', s=15, zorder=2)
    
    # 4. Mark Start and End points
    ax.plot(lons[0], lats[0], 
            'go', markersize=12, markerfacecolor='none', 
            markeredgewidth=3, label='Start (Oldest)', zorder=3)
    ax.plot(lons[-1], lats[-1], 
            'rs', markersize=12, markerfacecolor='none', 
            markeredgewidth=3, label='End (Newest)', zorder=3)
            
    # 5. Add a color bar to show what the colors mean
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
    
    # Convert timestamp ticks to readable format (datetime)
    cbar_ticks = cbar.get_ticks()
    # Convert Unix timestamps to datetime objects
    datetime_ticks = [datetime.fromtimestamp(t) for t in cbar_ticks]
    # Format as strings
    cbar.ax.set_yticklabels([dt.strftime('%Y-%m-%d\n%H:%M') for dt in datetime_ticks])
    cbar.set_label('Timestamp (UTC)', rotation=270, labelpad=20)

    # 6. Add labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Vessel Trajectory', fontsize=16, fontweight='bold')
    
    # 7. Final touches
    
    # --- Set fixed plot boundaries ---
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Use 'equal' axis scaling for a correct geographic representation
    # This adjusts the aspect ratio based on the mean latitude
    # to make distances on the map more representative.
    if np.cos(np.deg2rad(mean_lat)) > 0: # Avoid division by zero if lat is 90
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    else:
        ax.set_aspect('equal') # Default fallback
        
    # --- Add the basemap ---
    # We add this *after* setting limits and aspect ratio
    # crs='EPSG:4674' tells contextily our plot is in Lat/Lon
    # zorder=1 places the map *under* our data (which has zorder=2)
    # Stamen.TonerLite is no longer available. Using CartoDB.Positron as a replacement.
    # You can also try cx.providers.OpenStreetMap.Mapnik
    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron, zorder=1)
    
    plt.tight_layout()
    
    # 8. Show the plot
    print("Plot generated. Displaying...")
    plt.show()