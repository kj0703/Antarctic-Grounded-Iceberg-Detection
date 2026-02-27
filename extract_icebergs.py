"""
Iceberg Connected Component Extraction and Coastline Masking
This script extracts individual iceberg regions from binary masks and optionally applies a coastline mask.
"""

import os
import gc
import json
import re
import argparse
import psutil
from math import ceil
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

import numpy as np
import cv2
import rasterio
from rasterio.transform import xy
from rasterio.features import geometry_mask
import geopandas as gpd
from scipy.ndimage import binary_dilation

# ==========================================
# 1. Union-Find Algorithm for Boundary Merging
# ==========================================
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

# ==========================================
# 2. Coastline Masking (Pre-processing)
# ==========================================
def apply_coastline_mask(target_tif, output_tif, mask_gpkg, dilation_pixels=40):
    """Applies SCAR coastline mask to remove land and ice shelf instances."""
    mask_values = ['land', 'ice shelf', 'ice tongue', 'rumple']
    
    os.environ['GDAL_CACHEMAX'] = '256'
    os.environ['GDAL_NUM_THREADS'] = '1'

    try:
        with rasterio.open(target_tif) as src:
            data = src.read(1)
            meta = src.meta.copy()
            transform = src.transform
            crs = src.crs
            nodata = src.nodata or 0

        gdf = gpd.read_file(mask_gpkg, layer="add_coastline_high_res_polygon_v7_11")
        mask_gdf = gdf[gdf['surface'].isin(mask_values)].copy()

        if mask_gdf.empty:
            raise ValueError("No land or ice shelf features found in the provided GPKG.")

        if mask_gdf.crs != crs:
            mask_gdf = mask_gdf.to_crs(crs)

        mask_geom = mask_gdf.union_all()
        mask_bool = geometry_mask(
            geometries=[mask_geom],
            out_shape=data.shape,
            transform=transform,
            invert=True, 
            all_touched=True
        )

        if dilation_pixels > 0:
            mask_bool = binary_dilation(mask_bool, iterations=dilation_pixels)
        
        data[mask_bool] = nodata  

        meta.update({"driver": "GTiff", "compress": "deflate", "predictor": 1})
        with rasterio.open(output_tif, "w", **meta) as dst:
            dst.write(data, 1)

        return True

    except Exception as e:
        print(f"Failed to apply coastline mask to {target_tif}: {e}")
        return False

# ==========================================
# 3. Block-based Region Extraction (Core Logic)
# ==========================================
def extract_icebergs_block(args):
    """Extract connected components from TIFF mask blocks and filter out regions below the area threshold."""
    mask_block, timestamp, transform, block_row_offset, block_col_offset, min_area_threshold = args
    mask_block = (mask_block * 255).astype(np.uint8)
    num_labels, labeled_mask, stats, centroids = cv2.connectedComponentsWithStats(
        mask_block, connectivity=8
    )
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_threshold:
            continue
        pixel_centroid = centroids[i]
        row, col = (
            pixel_centroid[1] + block_row_offset,
            pixel_centroid[0] + block_col_offset,
        )
        geo_x, geo_y = xy(transform, row, col)
        region = {
            "area": float(area), 
            "pixel_centroid": [float(pixel_centroid[0] + block_col_offset), float(pixel_centroid[1] + block_row_offset)], 
            "geo_centroid": [float(geo_x), float(geo_y)],
            "label": int(i), 
            "bbox": [
                float(stats[i, cv2.CC_STAT_LEFT] + block_col_offset),
                float(stats[i, cv2.CC_STAT_TOP] + block_row_offset),
                float(stats[i, cv2.CC_STAT_WIDTH]),
                float(stats[i, cv2.CC_STAT_HEIGHT]),
            ], 
        }
        regions.append(region)
        
    # Filter regions smaller than the threshold
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_threshold:
            labeled_mask[labeled_mask == i] = 0
            
    return regions, labeled_mask, num_labels - 1, block_row_offset, block_col_offset

def merge_block_boundaries_with_tracking(labeled_mask, block_rows, block_cols, height, width):
    """Merge boundaries across blocks using a Union-Find approach with tracking."""
    block_height = ceil(height / block_rows)
    block_width = ceil(width / block_cols)
    max_label = int(np.max(labeled_mask))
    if max_label == 0:
        return labeled_mask, 0, [0], [0], {}, {}
    parent = list(range(max_label + 1))
    rank = [0] * (max_label + 1)
    merge_history = defaultdict(list) 
    print(f"Starting boundary merge with tracking for {max_label} labels...")
    
    # Horizontal boundaries
    for i in range(1, block_rows):
        boundary_row = i * block_height
        if boundary_row >= height:
            continue
        for col in range(width):
            # Directly adjacent
            if labeled_mask[boundary_row - 1, col] > 0 and labeled_mask[boundary_row, col] > 0:
                union(parent, rank, labeled_mask[boundary_row - 1, col], labeled_mask[boundary_row, col])
            # Diagonal left
            if col > 0:
                if labeled_mask[boundary_row - 1, col] > 0 and labeled_mask[boundary_row, col - 1] > 0:
                    union(parent, rank, labeled_mask[boundary_row - 1, col], labeled_mask[boundary_row, col - 1])
                if labeled_mask[boundary_row - 1, col - 1] > 0 and labeled_mask[boundary_row, col] > 0:
                    union(parent, rank, labeled_mask[boundary_row - 1, col - 1], labeled_mask[boundary_row, col])
            # Diagonal right
            if col < width - 1:
                if labeled_mask[boundary_row - 1, col] > 0 and labeled_mask[boundary_row, col + 1] > 0:
                    union(parent, rank, labeled_mask[boundary_row - 1, col], labeled_mask[boundary_row, col + 1])
                if labeled_mask[boundary_row - 1, col + 1] > 0 and labeled_mask[boundary_row, col] > 0:
                    union(parent, rank, labeled_mask[boundary_row - 1, col + 1], labeled_mask[boundary_row, col])
    
    # Vertical boundaries
    for j in range(1, block_cols):
        boundary_col = j * block_width
        if boundary_col >= width:
            continue
        for row in range(height):
            # Directly adjacent
            if labeled_mask[row, boundary_col - 1] > 0 and labeled_mask[row, boundary_col] > 0:
                union(parent, rank, labeled_mask[row, boundary_col - 1], labeled_mask[row, boundary_col])
            # Diagonal up
            if row > 0:
                if labeled_mask[row, boundary_col - 1] > 0 and labeled_mask[row - 1, boundary_col] > 0:
                    union(parent, rank, labeled_mask[row, boundary_col - 1], labeled_mask[row - 1, boundary_col])
                if labeled_mask[row - 1, boundary_col - 1] > 0 and labeled_mask[row, boundary_col] > 0:
                    union(parent, rank, labeled_mask[row - 1, boundary_col - 1], labeled_mask[row, boundary_col])
            # Diagonal down
            if row < height - 1:
                if labeled_mask[row, boundary_col - 1] > 0 and labeled_mask[row + 1, boundary_col] > 0:
                    union(parent, rank, labeled_mask[row, boundary_col - 1], labeled_mask[row + 1, boundary_col])
                if labeled_mask[row + 1, boundary_col - 1] > 0 and labeled_mask[row, boundary_col] > 0:
                    union(parent, rank, labeled_mask[row + 1, boundary_col - 1], labeled_mask[row, boundary_col])

    # Apply union and track merges
    print("Applying union and tracking merges...")
    for i in range(1, max_label + 1):
        find(parent, i)
        
    for row in range(height):
        for col in range(width):
            if labeled_mask[row, col] > 0:
                old_label = labeled_mask[row, col]
                new_label_root = parent[old_label]
                labeled_mask[row, col] = new_label_root
                if old_label != new_label_root:
                    merge_history[new_label_root].append(old_label)

    # Remap to continuous labels
    print("Remapping to continuous labels...")
    unique_labels = np.unique(labeled_mask[labeled_mask > 0])
    label_map = {old: new for new, old in enumerate(sorted(unique_labels), 1)}
    
    updated_merge_history = defaultdict(list)
    for old_root, merged_labels in merge_history.items():
        new_root = label_map.get(old_root, 0)
        if new_root == 0: continue
        
        all_related_pre_dsu = set(merged_labels)
        all_related_pre_dsu.add(old_root)
        related_roots = {parent[l] for l in all_related_pre_dsu}
        related_final_labels = {label_map.get(r, 0) for r in related_roots}
        for final_l in related_final_labels:
            if final_l > 0 and final_l != new_root:
                updated_merge_history[new_root].append(final_l)

    for row in range(height):
        for col in range(width):
            if labeled_mask[row, col] > 0:
                labeled_mask[row, col] = label_map.get(labeled_mask[row, col], 0)
                
    new_max_label = len(label_map)
    print(f"Boundary merge with tracking completed. New max label: {new_max_label}")
    return labeled_mask, new_max_label, parent, rank, label_map, updated_merge_history

def aggregate_regions_precise(
    labeled_mask, all_regions, parent, label_map, updated_merge_history,
    timestamp, transform, min_area_threshold
):
    """Aggregate regions using precise incremental calculation following boundary merges."""
    print("Aggregating regions using precise incremental calculation...")
    region_dict = {r["label"]: r for r in all_regions}
    merged_final_labels_set = set()
    for new_root, other_final_labels in updated_merge_history.items():
        merged_final_labels_set.add(new_root)
        merged_final_labels_set.update(other_final_labels)
    
    new_regions = []
    processed_old_roots = set()

    # Type A: Unmerged regions
    for pre_dsu_label, region in region_dict.items():
        old_root = find(parent, pre_dsu_label)
        final_label = label_map.get(old_root)
        if final_label and final_label not in merged_final_labels_set:
            if old_root not in processed_old_roots:
                region["label"] = final_label
                region["unique_id"] = f"{timestamp}_region_{final_label}"
                new_regions.append(region)
                processed_old_roots.add(old_root)
    print(f"Type A (unmerged) regions processed: {len(new_regions)}")

    # Type B: Merged regions
    for final_label_root in updated_merge_history.keys():
        temp_mask = (labeled_mask == final_label_root).astype(np.uint8)
        num_labels_temp, _, stats_temp, centroids_temp = cv2.connectedComponentsWithStats(temp_mask, connectivity=8)
        
        if num_labels_temp < 2:
            continue
            
        stats = stats_temp[1]
        centroid_px = centroids_temp[1]
        area = stats[cv2.CC_STAT_AREA]
        
        if area < min_area_threshold:
            continue
            
        px_col, px_row = centroid_px[0], centroid_px[1]
        geo_x, geo_y = xy(transform, px_row, px_col)
        bbox = [
            float(stats[cv2.CC_STAT_LEFT]), float(stats[cv2.CC_STAT_TOP]),
            float(stats[cv2.CC_STAT_WIDTH]), float(stats[cv2.CC_STAT_HEIGHT])
        ]
        new_region = {
            "area": float(area),
            "pixel_centroid": [round(px_col, 2), round(px_row, 2)],
            "geo_centroid": [round(geo_x, 6), round(geo_y, 6)],
            "label": int(final_label_root),
            "bbox": bbox,
            "unique_id": f"{timestamp}_region_{int(final_label_root)}",
            "merged_from": [int(l) for l in updated_merge_history[final_label_root]],
        }
        new_regions.append(new_region)
        
    print(f"Type B (merged) regions processed: {len(updated_merge_history.keys())}")
    print(f"Aggregation completed: {len(new_regions)} total regions")
    return sorted(new_regions, key=lambda r: r["label"])

def process_iceberg_extraction_block(
    base_mask_folder, timestamps, output_folder, block_rows, block_cols, min_area_threshold,
):
    """Main processing function to extract iceberg regions from the mask time series."""
    os.makedirs(output_folder, exist_ok=True)
    regions_list = []
    
    for timestamp in timestamps:
        mask_path = os.path.join(base_mask_folder, f"{timestamp}_mask.tif")
        out_mask = os.path.join(output_folder, f"labeled_mask_{timestamp}.tif")
        out_json = os.path.join(output_folder, f"regions_{timestamp}.json")
        
        if os.path.exists(out_mask) and os.path.exists(out_json):
            print(f"Skipped, already completed: {timestamp}")
            continue
            
        if not os.path.exists(mask_path):
            print(f"Mask file {mask_path} does not exist, skipping.")
            continue
            
        try:
            with rasterio.open(mask_path) as src:
                mask_data = src.read(1)
                if src.profile["dtype"] in ["uint8", "int8"]:
                    mask = mask_data / 255.0
                else:
                    mask = mask_data.astype(np.float32)
                mask = (mask > 0.5).astype(np.float32)
                transform = src.transform
                crs = src.crs
                height, width = mask.shape
                profile = src.profile
                print(f"Loaded mask {mask_path}: {height}x{width}, CRS {crs}")
        except Exception as e:
            print(f"Error reading {mask_path}: {e}")
            continue
            
        if np.sum(mask) == 0:
            print(f"Skipping {mask_path}: No positive pixels identified.")
            continue

        # Parallel block execution
        labeled_mask = np.zeros((height, width), dtype=np.uint32)
        all_regions = []
        block_height = ceil(height / block_rows)
        block_width = ceil(width / block_cols)
        tasks = []
        
        for i in range(block_rows):
            for j in range(block_cols):
                r0 = i * block_height
                r1 = min((i + 1) * block_height, height)
                c0 = j * block_width
                c1 = min((j + 1) * block_width, width)
                if r1 <= r0 or c1 <= c0: continue
                block = mask[r0:r1, c0:c1]
                tasks.append((block, timestamp, transform, r0, c0, min_area_threshold))
                
        num_processes = min(os.cpu_count(), max(4, os.cpu_count() // 2), len(tasks))
        print(f"Starting parallel extraction with {num_processes} processes...")
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(extract_icebergs_block, tasks)

        # Collect and remap results
        label_mapping = {}
        global_new_label = 1
        for idx, (regions, block_mask, n_labels, r0, c0) in enumerate(results):
            r1 = min(r0 + block_height, height)
            c1 = min(c0 + block_width, width)
            block_mapping = {}
            for region in regions:
                old_l = region["label"]
                key = (idx, old_l)
                if key not in label_mapping:
                    label_mapping[key] = global_new_label
                    region["label"] = global_new_label
                    global_new_label += 1
                else:
                    region["label"] = label_mapping[key]
                block_mapping[old_l] = region["label"]
            
            temp_block_mask = np.zeros_like(block_mask, dtype=np.uint32)
            for old_l, new_l in block_mapping.items():
                temp_block_mask[block_mask == old_l] = new_l
            labeled_mask[r0:r1, c0:c1] = temp_block_mask
            all_regions.extend(regions)
        
        del results, tasks, temp_block_mask, block_mask
        gc.collect()
        
        # Adjust profile dtype based on label count
        if global_new_label > 65535:
             profile.update(dtype=np.uint32)
             labeled_mask = labeled_mask.astype(np.uint32)
        else:
             profile.update(dtype=np.uint16)
             labeled_mask = labeled_mask.astype(np.uint16)

        # Merge boundaries across blocks
        labeled_mask, max_label, parent, rank, label_map, merge_history = merge_block_boundaries_with_tracking(
            labeled_mask, block_rows, block_cols, height, width
        )
        
        # Aggregate the final region statistics
        all_regions = aggregate_regions_precise(
            labeled_mask, all_regions, parent, label_map, merge_history,
            timestamp, transform, min_area_threshold
        )
        
        regions_list.append(all_regions)
        
        # Save output data
        profile.update(nodata=0, count=1, compress='deflate', predictor=2)
        with rasterio.open(out_mask, "w", **profile) as dst:
            dst.write(labeled_mask, 1)
        print(f"Successfully saved labeled mask to -> {out_mask}")
        
        with open(out_json, "w") as f:
            json.dump(all_regions, f, indent=2)
        print(f"Successfully saved regions JSON to -> {out_json}")
        
        del mask, labeled_mask, all_regions, parent, rank, label_map, merge_history
        gc.collect()
        
    return regions_list

# ==========================================
# 4. Main Execution Interface
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Extract iceberg connected components from binary SAR masks.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input binary mask TIFFs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save labeled masks and JSON region data")
    parser.add_argument("--coastline_gpkg", type=str, default=None, help="(Optional) Path to SCAR coastline GPKG for land masking")
    parser.add_argument("--min_area", type=int, default=10, help="Minimum area threshold (in pixels) for icebergs")
    parser.add_argument("--block_rows", type=int, default=4, help="Number of rows for block processing")
    parser.add_argument("--block_cols", type=int, default=4, help="Number of columns for block processing")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Optional Coastline Masking
    processed_dir = input_path
    if args.coastline_gpkg and os.path.exists(args.coastline_gpkg):
        print(f"Applying coastline mask from {args.coastline_gpkg}...")
        processed_dir = output_path
        
        tif_files = list(input_path.rglob("*_mask.tif"))
        for tif in tif_files:
            out_tif = processed_dir / tif.name
            if not out_tif.exists():
                apply_coastline_mask(tif, out_tif, args.coastline_gpkg)
    
    # Step 2: Extract Connected Regions
    print("Initiating connected region extraction process...")
    timestamps = []
    for f in os.listdir(processed_dir):
        match = re.search(r'(\d{8}T\d{6})_mask\.tif', f)
        if match:
            timestamps.append(match.group(1))
    
    timestamps = sorted(list(set(timestamps)))
    
    if not timestamps:
        print(f"No valid *_mask.tif files discovered in {processed_dir}.")
        return

    print(f"Discovered {len(timestamps)} chronological timestamps. Commencing processing...")
    
    process_iceberg_extraction_block(
        base_mask_folder=str(processed_dir),
        timestamps=timestamps,
        output_folder=str(output_path),
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        min_area_threshold=args.min_area
    )
    
    print("All extraction tasks have been completed successfully.")

if __name__ == "__main__":
    main()
