"""
Physical and Environmental Trajectory Filter
This script post-processes raw iceberg trajectories by applying physical area corrections 
(latitude-dependent scaling) and environmental filters (Bathymetry and Sea Ice Concentration) 
to eliminate false positives in deep water or high-density sea ice regions.
"""

import os
import argparse
import pandas as pd
import numpy as np
import xarray as xr
import pyproj
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from tqdm import tqdm

# ==========================================
# 1. Coordinate and Spatial Helpers
# ==========================================
def get_coord_transformer():
    """Initialise coordinate transformer from EPSG:3031 to OSI-SAF Polar Stereographic."""
    crs_src = pyproj.CRS.from_epsg(3031)
    proj_osisaf_str = "+proj=stere +a=6378273 +b=6356889.44891 +lat_0=-90 +lat_ts=-70 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    crs_target = pyproj.CRS.from_proj4(proj_osisaf_str)
    return pyproj.Transformer.from_crs(crs_src, crs_target, always_xy=True)

def get_mask_filename(timestamp_str, prefix="raw_"):
    try:
        dt = pd.to_datetime(timestamp_str)
        return f"{prefix}labeled_mask_{dt.strftime('%Y%m%dT%H%M%S')}.tif"
    except Exception:
        return None

def extract_label_from_uid(uid):
    """Extract the integer pixel label from a Unique_ID string."""
    try:
        return int(str(uid).split('_region_')[-1])
    except Exception:
        return -1

# ==========================================
# 2. Environmental Data Extraction
# ==========================================
def get_bathymetry_batch(x_3031_array, y_3031_array, bathy_path):
    """
    Extract bathymetry data by dynamically reprojecting IBCSO (EPSG:9354) to EPSG:3031 
    using WarpedVRT to ensure accurate spatial sampling.
    """
    default_res = np.zeros(len(x_3031_array))
    
    if not os.path.exists(bathy_path):
        print(f"  [Error] Bathymetry file not located: {bathy_path}")
        return default_res

    try:
        with rasterio.open(bathy_path) as src:
            dst_crs = 'EPSG:3031'
            with WarpedVRT(src, crs=dst_crs, resampling=Resampling.bilinear) as vrt:
                coords = list(zip(x_3031_array, y_3031_array))
                sampled_generator = vrt.sample(coords, indexes=1)
                sampled_values = np.array([val[0] for val in sampled_generator])
                
                if src.nodata is not None:
                     sampled_values = np.where(sampled_values == src.nodata, np.nan, sampled_values)
                return sampled_values
                
    except Exception as e:
        print(f"  [Error] Failed to sample bathymetry via WarpedVRT: {e}")
        return default_res

def get_sic_value_batch(date_str, x_3031_array, y_3031_array, transformer, sic_dir):
    """Extract Sea Ice Concentration (SIC) values from OSI-SAF NetCDF files."""
    date_part = str(date_str).split('T')[0] if 'T' in str(date_str) else str(date_str)[:8]
    filename = f"ice_conc_sh_polstere-100_amsr2_{date_part}1200.nc"
    file_path = os.path.join(sic_dir, filename)
    default_res = np.zeros(len(x_3031_array))

    if not os.path.exists(file_path):
        return default_res

    try:
        x_m, y_m = transformer.transform(x_3031_array, y_3031_array)
        x_km = x_m / 1000.0
        y_km = y_m / 1000.0
        
        with xr.open_dataset(file_path, engine='netcdf4', chunks={}) as ds:
            da = ds['ice_conc'].squeeze(drop=True)
            x_indexer = xr.DataArray(x_km, dims="points")
            y_indexer = xr.DataArray(y_km, dims="points")
            sic_values = da.sel(xc=x_indexer, yc=y_indexer, method='nearest', tolerance=20).values
            
            if sic_values.ndim > 1: sic_values = sic_values.flatten()
            sic_values = np.nan_to_num(sic_values, nan=0.0)
            
            if len(sic_values) != len(x_3031_array):
                return default_res
            return sic_values
    except Exception:
        return default_res

# ==========================================
# 3. Physical Area Correction
# ==========================================
def calculate_areas_sparse_by_uid(tiff_path, target_uids):
    """
    Perform physics-based area correction using absolute latitude to determine the 
    accurate scaling factor (k) for EPSG:3031 projection distortion.
    """
    results = {uid: {'area_km2': 0.0, 'pixels': 0} for uid in target_uids}
    proj_3031 = pyproj.Proj("EPSG:3031")
    # Standard parallel for EPSG:3031 is 71 degrees South
    phi_std = np.radians(71) 
    
    try:
        with rasterio.open(tiff_path) as src:
            mask_data = src.read(1)
            transform = src.transform
            res_x, res_y = src.res
            map_pixel_area = abs(res_x * res_y)
            
            rows, cols = np.where(mask_data > 0)
            if len(rows) == 0:
                return results
            
            pixel_labels = mask_data[rows, cols].astype(np.int64)
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
            _, lats = proj_3031(np.array(xs), np.array(ys), inverse=True)
            
            # Absolute latitude for scaling factor calculation
            phi = np.abs(np.radians(lats))
            k = (1 + np.sin(phi)) / (1 + np.sin(phi_std))
            
            # True area conversion (square metres to square kilometres)
            pixel_real_areas = (map_pixel_area / (k**2)) / 1e6 
            
            max_label = pixel_labels.max()
            sum_areas = np.bincount(pixel_labels, weights=pixel_real_areas, minlength=max_label+1)
            sum_pixels = np.bincount(pixel_labels, minlength=max_label+1)
            
            for uid in target_uids:
                label = extract_label_from_uid(uid)
                if 0 < label < len(sum_areas):
                    results[uid]['area_km2'] = sum_areas[label]
                    results[uid]['pixels'] = int(sum_pixels[label])
                    
    except Exception as e:
        print(f"  [Error] Failed to read or process TIF {os.path.basename(tiff_path)}: {e}")
        for uid in target_uids: 
            results[uid] = {'area_km2': np.nan, 'pixels': np.nan}
            
    return results

# ==========================================
# 4. Main Processing Pipeline
# ==========================================
def process_orbit_pipeline(folder_path, folder_name, args):
    """Execute the complete post-processing pipeline for a single orbital folder."""
    input_csv = os.path.join(folder_path, f"Grounded_iceberg_tracks_{folder_name}.csv")
    output_csv = os.path.join(folder_path, f"Grounded_iceberg_tracks_{folder_name}_Final.csv")
    
    if not os.path.exists(input_csv):
        return
        
    if os.path.exists(output_csv):
        print(f"  [Skip] Orbit {folder_name} already contains a Final output. Skipping...")
        return
        
    print(f"\nProcessing Orbit: {folder_name}")
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"  [Notice] Dataset for {folder_name} is empty. Skipping.")
        return

    # ---------------------------------------------------------
    # Step 1: Track Filtering (Remove single-epoch anomalies)
    # ---------------------------------------------------------
    original_tracks = df['Track_ID'].nunique()
    track_counts = df.groupby('Track_ID')['Track_ID'].transform('count')
    df = df[track_counts >= 2].copy()
    new_tracks = df['Track_ID'].nunique()
    print(f"  [Step 1] Track Filtration: Retained {new_tracks}/{original_tracks} tracks (Removed {original_tracks - new_tracks} single-epoch anomalies).")

    if df.empty:
        print(f"  [Notice] No valid trajectories remained post-filtration. Skipping.")
        return

    # ---------------------------------------------------------
    # Step 2: Physical Area Correction
    # ---------------------------------------------------------
    df['Area_km2'] = np.nan
    df['Area_Pixels'] = 0
    grouped = df.groupby('Timestamp')
    
    prefix = f"{folder_name}_" 
    
    for timestamp, group in tqdm(grouped, desc="  [Step 2] Spatial Area Integration", leave=False):
        tif_name = get_mask_filename(timestamp, prefix=prefix) 
        if not tif_name: continue
        
        tif_path = os.path.join(folder_path, tif_name)
        if not os.path.exists(tif_path):
            continue
            
        uids = group['Unique_ID'].tolist()
        area_map = calculate_areas_sparse_by_uid(tif_path, uids)
        
        for uid in uids:
            res = area_map.get(uid)
            idx = (df['Timestamp'] == str(timestamp)) & (df['Unique_ID'] == uid)
            df.loc[idx, 'Area_km2'] = res['area_km2']
            df.loc[idx, 'Area_Pixels'] = res['pixels']

    # Generate geographical coordinates if absent
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        proj_3031 = pyproj.Proj("EPSG:3031")
        df['Longitude'], df['Latitude'] = proj_3031(df['Geo_X'].values, df['Geo_Y'].values, inverse=True)

    # ---------------------------------------------------------
    # Step 3: Area Pixel Filtration
    # ---------------------------------------------------------
    bad_pixel_tracks = df.loc[df['Area_Pixels'] < args.min_pixels, 'Track_ID'].unique()
    if len(bad_pixel_tracks) > 0:
        df = df[~df['Track_ID'].isin(bad_pixel_tracks)].copy()
    print(f"  [Step 3] Pixel Filtration: Removed {len(bad_pixel_tracks)} tracks containing frames with < {args.min_pixels} pixels.")

    if df.empty:
        print(f"  [Notice] No trajectories survived pixel filtration. Skipping.")
        return

    # ---------------------------------------------------------
    # Step 4: Environmental Extraction (Bathymetry & SIC)
    # ---------------------------------------------------------
    print(f"  [Step 4] Environmental Extraction: Fetching IBCSO Bathymetry (WarpedVRT)...")
    df['Bed_Depth'] = get_bathymetry_batch(df['Geo_X'].to_numpy(), df['Geo_Y'].to_numpy(), args.bathy_file)
    
    df['SIC_Value'] = 0.0
    transformer = get_coord_transformer()
    unique_timestamps = df['Timestamp'].unique()
    
    for ts in tqdm(unique_timestamps, desc="  Fetching OSI-SAF SIC", leave=False):
        mask = df['Timestamp'] == ts
        if not mask.any(): continue
        x_coords = df.loc[mask, 'Geo_X'].to_numpy()
        y_coords = df.loc[mask, 'Geo_Y'].to_numpy()
        sic_vals = get_sic_value_batch(ts, x_coords, y_coords, transformer, args.sic_dir)
        try:
            df.loc[mask, 'SIC_Value'] = sic_vals
        except ValueError:
            continue

    # ---------------------------------------------------------
    # Step 5: Environmental Quality Control Filtering
    # ---------------------------------------------------------
    # Condition 1: Deep water false positives
    cond_1 = (df['Area_km2'] < args.cond1_area) & (df['Bed_Depth'] < args.cond1_depth) & (df['SIC_Value'] > args.cond1_sic)
    # Condition 2: High-density sea ice / mélange
    cond_2 = (df['Area_km2'] < args.cond2_area) & (df['SIC_Value'] > args.cond2_sic)
    # Condition 3: Extreme deep water anomaly
    cond_3 = (df['Bed_Depth'] < args.cond3_depth)
    
    df['is_bad_frame'] = cond_1 | cond_2 | cond_3
    
    track_bad_counts = df.groupby('Track_ID')['is_bad_frame'].sum()
    tracks_to_remove = track_bad_counts[track_bad_counts >= args.min_bad_frames].index
    
    df_clean = df[~df['Track_ID'].isin(tracks_to_remove)].copy()
    df_clean.drop(columns=['is_bad_frame'], inplace=True, errors='ignore')

    print(f"  [Step 5] Environmental QC: Removed {len(tracks_to_remove)} tracks violating depth or SIC parameters.")

    # ---------------------------------------------------------
    # Output Generation
    # ---------------------------------------------------------
    target_cols = [
        'Track_ID', 'Timestamp', 'Area_km2', 'Area_Pixels', 
        'Latitude', 'Longitude', 'Geo_X', 'Geo_Y', 
        'Unique_ID', 'Bed_Depth', 'SIC_Value'
    ]
    existing_cols = [c for c in target_cols if c in df_clean.columns]
    df_final = df_clean[existing_cols].copy()
    
    df_final.to_csv(output_csv, index=False)
    print(f"  Successfully saved refined dataset to: {os.path.basename(output_csv)}")

# ==========================================
# 5. CLI Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Physical and Environmental Post-Processing Filter.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the tracking CSV.")
    parser.add_argument("--bathy_file", type=str, required=True, help="Path to IBCSO bathymetry.")
    parser.add_argument("--sic_dir", type=str, required=True, help="Directory containing OSI-SAF SIC files.")
    
    # Thresholds (Defaults for IW mode)
    parser.add_argument("--min_pixels", type=int, default=40)
    parser.add_argument("--cond1_area", type=float, default=2500.0)
    parser.add_argument("--cond1_depth", type=float, default=-600.0)
    parser.add_argument("--cond1_sic", type=float, default=40.0)
    parser.add_argument("--cond2_area", type=float, default=250.0)
    parser.add_argument("--cond2_sic", type=float, default=80.0)
    parser.add_argument("--cond3_depth", type=float, default=-1000.0)
    parser.add_argument("--min_bad_frames", type=int, default=1)
    
    args = parser.parse_args()

    root_dir = args.input_dir
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} not found.")
        return

    if os.path.exists(os.path.join(root_dir, "Grounded_iceberg_tracks_coastline.csv")):
        folder_name = "coastline"
    elif os.path.exists(os.path.join(root_dir, "Grounded_iceberg_tracks_raw.csv")):
        folder_name = "raw"
    else:
        print(f"Error: No tracking CSV (coastline or raw) found in {root_dir}.")
        return

    print(f"Initiating Post-Processing Pipeline for: {folder_name} data")
    
    try:
        process_orbit_pipeline(root_dir, folder_name, args)
    except Exception as e:
        print(f"  [Critical Error] Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
