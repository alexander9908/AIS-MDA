import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

STYLE = {
    "past": "#1f77b4", "true": "#2ca02c", 
    "land": "#E0E0E0", "edge": "#505050", "water": "#FFFFFF", "grid": "#B0B0B0"
}

plt.rcParams.update({
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333", "font.family": "sans-serif", 
    "axes.facecolor": STYLE["water"]
})

def conform_data(d_array):
    return {
        'lats_past': d_array['past'][:,0],
        'lons_past': d_array['past'][:,1],
        'lats_true': d_array['true'][:,0],
        'lons_true': d_array['true'][:,1],
        'lats_pred': d_array['pred'][:,0],
        'lons_pred': d_array['pred'][:,1],
    }

def plot_trajectories(mmsi,
                      time,
                      data,
                      output_dir=None,
                      title=None,
                      show=True,
                      fig_size=(10,8)):
    """
    Plots past, true, and predicted trajectories for one or more models on a map.
    
    Parameters:
    - mmsi: int
    - time: int
    - data: dict of 'past', 'true', 'pred' each with numpy arrays of shape (T, 2) or dict of {model_name: data_dict}
    - output_dir: str or Path, directory to save the plot. If None, the plot is not saved.
    - title: str, optional title for the plot. If None, no title is set.
    - show: bool, whether to display the plot. Default is True.
    - fig_size: tuple, size of the figure. Default is (10, 8).
    
    Example usage:
    plot_trajectories(
        1234,
        1609459200,
        {
            'ModelA': {
                'past': np.array([[...], [...], ...]),
                'true': np.array([[...], [...], ...]),
                'pred': np.array([[...], [...], ...])
            },
            'ModelB': {
                'past': np.array([[...], [...], ...]),
                'true': np.array([[...], [...], ...]),
                'pred': np.array([[...], [...], ...])
            }
        },
        output_dir='plots',
        title='Vessel Trajectories on 2021-01-01',
        show=True
    )
    """
    
    if 'past' in data.keys(): # assuming single model
        preds = {'Pred': conform_data(data)}
    elif isinstance(data, dict):
        preds = {k: conform_data(v) for k, v in data.items()}
    else:
        raise ValueError("Data must be either a single trajectory dict or a dict of model trajectories.")

    model_names = list(preds.keys())
    ref_key = model_names[0]
    ref_data = preds[ref_key]

    lats_past = ref_data['lats_past']
    lons_past = ref_data['lons_past']
    lats_true = ref_data['lats_true']
    lons_true = ref_data['lons_true']

    for k, v in preds.items():
        if k == ref_key: continue
        if not (np.allclose(v['lats_past'], lats_past) and 
                np.allclose(v['lons_past'], lons_past) and
                np.allclose(v['lats_true'], lats_true) and
                np.allclose(v['lons_true'], lons_true)):
            print(f"Warning: Model '{k}' has mismatching Past/True data compared to '{ref_key}'.")

    all_lats_list = [lats_past, lats_true]
    all_lons_list = [lons_past, lons_true]
    
    for v in preds.values():
        all_lats_list.append(v['lats_pred'])
        all_lons_list.append(v['lons_pred'])

    all_lats = np.concatenate(all_lats_list)
    all_lons = np.concatenate(all_lons_list)
    
    pad = 0.1
    p_lat_min, p_lat_max = np.min(all_lats)-pad, np.max(all_lats)+pad
    p_lon_min, p_lon_max = np.min(all_lons)-pad, np.max(all_lons)+pad
    
    lat_range = p_lat_max - p_lat_min
    lon_range = p_lon_max - p_lon_min
    lat_center = (p_lat_min + p_lat_max) / 2
    lon_center = (p_lon_min + p_lon_max) / 2

    mercator_scale = np.cos(np.radians(lat_center))
    
    desired_aspect = fig_size[0] / fig_size[1]
    current_visual_aspect = (lon_range * mercator_scale) / lat_range
    
    if current_visual_aspect > desired_aspect:
        # Too wide: Increase Latitude range
        new_lat_range = (lon_range * mercator_scale) / desired_aspect
        p_lat_min = lat_center - new_lat_range / 2
        p_lat_max = lat_center + new_lat_range / 2
    else:
        # Too tall: Increase Longitude range
        new_lon_range = (lat_range * desired_aspect) / mercator_scale
        p_lon_min = lon_center - new_lon_range / 2
        p_lon_max = lon_center + new_lon_range / 2

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([p_lon_min, p_lon_max, p_lat_min, p_lat_max], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.GSHHSFeature(scale='h', levels=[1], 
                   facecolor=STYLE["land"], edgecolor=STYLE["edge"]))
    
    # Plot Shared Data
    ax.plot(lons_past, lats_past, transform=ccrs.PlateCarree(), 
            c=STYLE["past"], lw=2, label="Past")
    ax.plot(lons_true, lats_true, transform=ccrs.PlateCarree(), 
            c=STYLE["true"], lw=3, label="True")
    
    # Plot Predictions
    cmap = plt.get_cmap("tab10")
    for i, (model_name, d_mod) in enumerate(preds.items()):
        # Offset color by 3 to avoid blue(0) and green(2)
        color = cmap((i + 3) % 10)
        ax.plot(d_mod['lons_pred'], d_mod['lats_pred'], 
                transform=ccrs.PlateCarree(), 
                c=color, lw=3, ls="--", label=model_name)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color=STYLE['grid'], alpha=0.3, linestyle='--')
    
    gl.top_labels = False
    gl.right_labels = False
    
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': '#333333'}
    gl.ylabel_style = {'size': 10, 'color': '#333333'}

    ax.legend(loc='upper left')
    
    if title is not None:
        ax.set_title(title)
    
    if output_dir is not None:
        outdir = Path(output_dir) / str(mmsi)
        outdir.mkdir(parents=True, exist_ok=True)
        if len(preds) == 1:
            m_name = list(preds.keys())[0]
            fname = outdir / f"traj_{m_name}_{mmsi}_{time}.png"
        else:
            fname = outdir / f"traj_compare_{mmsi}_{time}.png"
            
        fig.savefig(fname, bbox_inches='tight', dpi=300)
        
    if show:
        plt.show()
    plt.close(fig)