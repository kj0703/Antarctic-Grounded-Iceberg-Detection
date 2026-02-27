"""
Geospatial Dataset Builder, Deduplicator, and Environment Classifier
This script transforms the filtered trajectory CSV data into a fully formatted GeoPackage (GPKG) dataset.
It extracts the precise representative polygon for each iceberg, resolves spatial overlaps (deduplication) 
across adjacent orbital passes, and performs a topological classification against a reference Fast Ice dataset.
"""

import sys
import os
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Patch for legacy numpy matrix dependency in certain Geopandas/Shapely environments
if not hasattr(np, "matrix"):
    class dummy_matrix(np.ndarray):
        pass
    np.matrix = dummy_matrix
    sys.modules["numpy"].matrix = dummy_matrix

# ==========================================
# 1. Geometry Extraction Functions
# ==========================================
def get_tif_path(row, mask_dir, prefix=""):
    """Construct the absolute path to the respective TIF mask in the flattened directory."""
    try:
        filename = f"{prefix}labeled_mask_{row['Timestamp']}.tif"
        full_path = os.path.join(mask_dir, filename)
        
        # Fallback for raw files without prefix
        if not os.path.exists(full_path):
            fallback = os.path.join(mask_dir, f"labeled_mask_{row['Timestamp']}.tif")
            if os.path.exists(fallback):
                return fallback
                
        return full_path
    except Exception:
        return None

def extract_geometry(task_data):
    """Vectorise the binary mask of a specific iceberg to generate its polygon geometry."""
    uid, row_dict = task_data
    tif_path = row_dict['tif_path']
    geo_x = row_dict['Geo_X']
    geo_y = row_dict['Geo_Y']
    
    if not os.path.exists(tif_path):
        return uid, None, "File Not Found"

    try:
        with rasterio.open(tif_path) as src:
            row_idx, col_idx = src.index(geo_x, geo_y)
            
            if (row_idx < 0 or row_idx >= src.height or col_idx < 0 or col_idx >= src.width):
                return uid, None, "Coordinates Out of Bounds"

            label_id = src.read(1, window=((row_idx, row_idx+1), (col_idx, col_idx+1)))[0, 0]
            
            if label_id == 0:
                return uid, None, "Background Label Detected"

            mask_data = src.read(1)
            binary_mask = (mask_data == label_id).astype('uint8')
            
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(shapes(binary_mask, mask=binary_mask, transform=src.transform))
            )
            
            geoms = list(results)
            if not geoms:
                return uid, None, "Vectorisation Failed"
            
            poly_shape = shape(geoms[0]['geometry'])
            return uid, poly_shape, "Success"
            
    except Exception as e:
        return uid, None, f"Error: {str(e)}"

def process_single_orbit(orbit_name, df_orbit, col_map, output_path, mask_dir, max_workers, prefix):
    """Extract and aggregate representative polygons for all icebergs within a specific orbit."""
    id_col = col_map['id']
    pixel_col = col_map['pixel']
    x_col = col_map['x']
    y_col = col_map['y']
    mode_col = col_map['mode']

    agg_funcs = {
        'Timestamp': ['first', 'last', 'count'],
        'Area_km2': ['mean', 'min', 'max'],
        pixel_col: ['mean', 'min', 'max'],
        mode_col: 'first',
        'Orbit': 'first',
        'Bed_Depth': 'mean' if 'Bed_Depth' in df_orbit.columns else 'first',
    }
    
    grouped = df_orbit.sort_values('Timestamp').groupby(id_col)
    summary = grouped.agg(agg_funcs)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    final_data = []
    extraction_tasks = []

    for idx, row in summary.iterrows():
        uid = row[id_col]
        try:
            original_group = grouped.get_group(uid)
            mid_idx = len(original_group) // 2
            mid_row = original_group.iloc[mid_idx]
            
            tif_path = get_tif_path(mid_row, mask_dir, prefix)
            
            date_range = f"{row['Timestamp_first']} - {row['Timestamp_last']}"
            area_range_km2 = f"{row['Area_km2_min']:.3f} - {row['Area_km2_max']:.3f}"
            area_range_px = f"{int(row[f'{pixel_col}_min'])} - {int(row[f'{pixel_col}_max'])}"

            bed_depth_val = row['Bed_Depth_mean'] if 'Bed_Depth_mean' in row else row.get('Bed_Depth_first', np.nan)

            record = {
                'Global_UID': uid,
                'Orbit': str(row['Orbit_first']),
                'Timestamp': mid_row['Timestamp'],
                'Date_Range': date_range,
                'Area_Mean_km2': row['Area_km2_mean'],
                'Area_Range_km2': area_range_km2,
                'Area_Mean_px': row[f'{pixel_col}_mean'],
                'Area_Range_px': area_range_px,
                'Bed_Depth': bed_depth_val,
                'Swath_Mode': row[f'{mode_col}_first'],
                'Latitude': mid_row.get('Latitude', np.nan),
                'Longitude': mid_row.get('Longitude', np.nan),
                'Northing_3031': mid_row[y_col],
                'Easting_3031': mid_row[x_col],
            }
            final_data.append(record)
            
            if tif_path:
                task_info = {'tif_path': tif_path, 'Geo_X': mid_row[x_col], 'Geo_Y': mid_row[y_col]}
                extraction_tasks.append((uid, task_info))
                
        except Exception:
            continue

    if not extraction_tasks:
        return False

    geom_dict = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_geometry, task) for task in extraction_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Extracting Polygons", leave=False):
            uid, poly, status = future.result()
            if poly:
                geom_dict[uid] = poly

    valid_records = []
    for rec in final_data:
        uid = rec['Global_UID']
        if uid in geom_dict:
            rec['geometry'] = geom_dict[uid]
            valid_records.append(rec)
    
    if not valid_records:
        return False

    gdf = gpd.GeoDataFrame(valid_records, crs="EPSG:3031")
    
    columns_order = [
        'Global_UID', 'Orbit', 'Timestamp', 'Date_Range', 
        'Area_Mean_km2', 'Area_Range_km2', 'Area_Mean_px', 'Area_Range_px', 
        'Bed_Depth', 'Swath_Mode', 'Latitude', 'Longitude', 
        'Northing_3031', 'Easting_3031', 'geometry'
    ]
    cols_to_use = [c for c in columns_order if c in gdf.columns]
    gdf = gdf[cols_to_use]
    
    gdf.to_file(output_path, driver="GPKG")
    print(f"  [Saved] Compiled {len(gdf)} spatial records for Orbit {orbit_name}.")
    return True

# ==========================================
# 2. Spatial Deduplication
# ==========================================
def remove_spatial_duplicates(gdf, iou_threshold):
    """
    Remove exact spatial duplicates using R-Tree bounding box rough-filtering 
    and exact geometry intersection. Retains the largest iceberg polygon.
    """
    print(f"\nInitiating spatial deduplication (Threshold > {iou_threshold*100}%). Initial features: {len(gdf)}")
    if len(gdf) <= 1:
        return gdf

    # Sort by descending area and reset index to ensure alignment with spatial index
    gdf = gdf.sort_values(by='Area_Mean_km2', ascending=False).reset_index(drop=True)
    
    # Construct R-Tree spatial index
    sindex = gdf.sindex
    drop_indices = set()

    for i, row in tqdm(gdf.iterrows(), total=len(gdf), desc="  Evaluating overlaps", leave=False):
        if i in drop_indices:
            continue
            
        geom_main = row.geometry
        
        # Step 1: Bounding Box filter (Rapid isolation)
        possible_matches = list(sindex.query(geom_main, predicate='intersects'))
        
        for match_idx in possible_matches:
            if match_idx == i or match_idx in drop_indices:
                continue
                
            geom_candidate = gdf.geometry.iloc[match_idx]
            
            # Step 2: Exact intersection calculation
            if not geom_main.intersects(geom_candidate):
                continue

            intersection = geom_main.intersection(geom_candidate)
            if intersection.is_empty:
                continue

            candidate_area = geom_candidate.area
            if candidate_area == 0:
                continue
                
            # Calculate proportion of the smaller candidate covered by the main iceberg
            overlap_ratio = intersection.area / candidate_area
            
            if overlap_ratio > iou_threshold:
                drop_indices.add(match_idx)

    clean_gdf = gdf.drop(index=list(drop_indices))
    print(f"  [Cleaned] Identified and removed {len(drop_indices)} duplicates. Final feature count: {len(clean_gdf)}\n")
    return clean_gdf

# ==========================================
# 3. Fast Ice Spatial Classification
# ==========================================
def classify_fast_ice_status(iceberg_gdf, fast_ice_path):
    """Determine spatial topological relationship between icebergs and landfast ice."""
    print(f"Loading Fast Ice Reference Data from: {os.path.basename(fast_ice_path)}...")
    fast_ice_raw = gpd.read_file(fast_ice_path)
    
    fast_ice_polys = fast_ice_raw[fast_ice_raw['class_id'] == 1].copy()
    if len(fast_ice_polys) == 0:
        print("  [Error] No valid Class 1 features located in the Fast Ice GPKG.")
        return iceberg_gdf

    if fast_ice_polys.crs != iceberg_gdf.crs:
        print(f"  Reprojecting Fast Ice dataset to {iceberg_gdf.crs.to_string()}...")
        fast_ice_polys = fast_ice_polys.to_crs(iceberg_gdf.crs)
    
    print("  Executing spatial union on Fast Ice geometries (This may take a moment)...")
    fast_ice_union = unary_union(fast_ice_polys.geometry)
    
    print("  Evaluating topological intersections and containment...")
    col_name = 'Fast_Ice_Overlap_Status'
    iceberg_gdf[col_name] = 'Outside'
    
    intersects_mask = iceberg_gdf.geometry.intersects(fast_ice_union)
    within_mask = iceberg_gdf.geometry.within(fast_ice_union)
    
    iceberg_gdf.loc[intersects_mask, col_name] = 'Partial'
    iceberg_gdf.loc[within_mask, col_name] = 'Inside'
    
    print("\n" + "="*40)
    print(" Fast Ice Classification Summary ")
    print("="*40)
    print(iceberg_gdf[col_name].value_counts())
    print("="*40 + "\n")
    
    return iceberg_gdf

# ==========================================
# 4. Main Execution Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Construct Geospatial Dataset, Deduplicate, and Perform Environmental Classification.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the integrated trajectory CSV.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing the extracted TIF masks.")
    parser.add_argument("--fast_ice_gpkg", type=str, help="Optional: Path to the Fast Ice reference GPKG.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final compiled GPKG files.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="Overlap threshold (0.0 - 1.0) for removing smaller spatial duplicates.")
    parser.add_argument("--max_workers", type=int, default=6, help="Maximum number of parallel extraction processes.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "temp_orbits")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Loading trajectory dataset: {os.path.basename(args.input_csv)}...")
    df = pd.read_csv(args.input_csv)
    
    col_map = {
        'id': next((c for c in df.columns if c in ['Track_ID', 'Global_UID', 'ID']), 'Global_UID'),
        'pixel': next((c for c in df.columns if c.startswith('Area_Pix')), 'Area_Pixels'),
        'x': next((c for c in df.columns if c in ['Geo_X', 'Centroid_X', 'Easting_3031']), 'Geo_X'),
        'y': next((c for c in df.columns if c in ['Geo_Y', 'Centroid_Y', 'Northing_3031']), 'Geo_Y'),
        'mode': 'Swath Mode' if 'Swath Mode' in df.columns else 'Swath_Mode'
    }
    
    print(f"Detected Schema: ID={col_map['id']}, Pixels={col_map['pixel']}, X={col_map['x']}, Y={col_map['y']}")

    df['Timestamp'] = df['Timestamp'].astype(str)
    
    if 'Orbit' not in df.columns:
        print("  [Note] 'Orbit' column not found in CSV. Assigning default value: 'Unknown'")
        df['Orbit'] = 'Unknown'
    df['Orbit'] = df['Orbit'].astype(str)
    
    if col_map['mode'] not in df.columns:
        print(f"  [Note] '{col_map['mode']}' column missing. Defaulting to 'EW'.")
        df[col_map['mode']] = 'EW'

    unique_orbits = df['Orbit'].unique()
    
    csv_name = os.path.basename(args.input_csv)
    prefix = "coastline_" if "coastline" in csv_name.lower() else "raw_" if "raw" in csv_name.lower() else ""
    
    print(f"Commencing geometry extraction for {len(unique_orbits)} distinct orbital passes (Prefix: {prefix})...")

    orbit_gpkgs = []
    for orbit in tqdm(unique_orbits, desc="Overall Progress"):
        safe_orbit_name = orbit.replace("(", "_").replace(")", "")
        output_path = os.path.join(temp_dir, f"Orbit_{safe_orbit_name}.gpkg")
        
        if not os.path.exists(output_path):
            df_orbit = df[df['Orbit'] == orbit].copy()
            success = process_single_orbit(safe_orbit_name, df_orbit, col_map, output_path, args.mask_dir, args.max_workers, prefix)
            if not success:
                continue
        orbit_gpkgs.append(output_path)

    print("\nMerging orbital datasets into a unified master spatial dataframe...")
    gdf_list = [gpd.read_file(gpkg) for gpkg in orbit_gpkgs if os.path.exists(gpkg)]
    
    if not gdf_list:
        print("Error: No valid spatial features were extracted across any orbit. Terminating.")
        return
        
    master_gdf = pd.concat(gdf_list, ignore_index=True)

    # ---> Apply Spatial Deduplication here <---
    master_gdf = remove_spatial_duplicates(master_gdf, args.iou_threshold)

    if args.fast_ice_gpkg and os.path.exists(args.fast_ice_gpkg):
        master_gdf = classify_fast_ice_status(master_gdf, args.fast_ice_gpkg)
        
        layered_out = os.path.join(args.output_dir, "Grounded_Icebergs_Dataset_Layered.gpkg")
        print(f"Saving stratified architectural layers to: {layered_out}")
        for status in master_gdf['Fast_Ice_Overlap_Status'].unique():
            subset = master_gdf[master_gdf['Fast_Ice_Overlap_Status'] == status]
            subset.to_file(layered_out, layer=status, driver="GPKG")

    final_out = os.path.join(args.output_dir, "Grounded_Icebergs_Dataset_Final.gpkg")
    print(f"Saving integrated master dataset to: {final_out}")
    master_gdf.to_file(final_out, driver="GPKG")
    
    print("Geospatial dataset synthesis completed successfully.")

if __name__ == "__main__":
    main()
