"""
Identify Stationaries (Iceberg Tracking Algorithm)
This script tracks individual grounded icebergs across multi-temporal sequences using a 
cost-matrix optimisation approach based on the Hungarian algorithm. It evaluates 
centroid distance, shape (contours), area, and Intersection over Union (IoU).
"""

import os
import json
import time
import argparse
import numpy as np
import cv2
import pandas as pd
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import rasterio
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# ==========================================
# 1. Feature Extraction and Geometry
# ==========================================
def compute_iou(mask1, mask2, bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) of two masks based on their bounding box intersection."""
    left1, top1, w1, h1 = [int(x) for x in bbox1]
    left2, top2, w2, h2 = [int(x) for x in bbox2]
   
    left = max(left1, left2)
    top = max(top1, top2)
    right = min(left1 + w1, left2 + w2)
    bottom = min(top1 + h1, top2 + h2)
    width = max(0, right - left)
    height = max(0, bottom - top)
   
    if width <= 0 or height <= 0:
        return 0.0
   
    mask1_region = mask1[max(0, top - top1):max(0, top - top1) + height,
                         max(0, left - left1):max(0, left - left1) + width]
    mask2_region = mask2[max(0, top - top2):max(0, top - top2) + height,
                         max(0, left - left2):max(0, left - left2) + width]
   
    if mask1_region.shape != mask2_region.shape:
        min_height = min(mask1_region.shape[0], mask2_region.shape[0])
        min_width = min(mask1_region.shape[1], mask2_region.shape[1])
        mask1_region = mask1_region[:min_height, :min_width]
        mask2_region = mask2_region[:min_height, :min_width]
   
    intersection = np.logical_and(mask1_region, mask2_region).sum()
    union = np.logical_or(mask1_region, mask2_region).sum()
    if union == 0:
        return 0
    return intersection / union

def get_axis_ratio(contour):
    """Calculate the major to minor axis ratio of an iceberg's contour."""
    if contour is None or len(contour) < 5:
        return 1.0
    ellipse = cv2.fitEllipse(contour)
    (center, axes, angle) = ellipse
    major = max(axes)
    minor = max(min(axes), 1e-5)
    ratio = major / minor
    return ratio

def smooth_contour(contour, ksize=5):
    """Apply moving average to smooth the contour, mitigating pixelation noise."""
    if contour is None:
        return None
    contour = contour.reshape(-1, 2)
    if len(contour) < ksize:
        return contour.reshape(-1, 1, 2)
    kernel = np.ones(ksize) / ksize
    xs = np.convolve(contour[:, 0], kernel, mode='same')
    ys = np.convolve(contour[:, 1], kernel, mode='same')
    smooth = np.vstack((xs, ys)).T.astype(np.int32)
    return smooth.reshape(-1, 1, 2)

# ==========================================
# 2. Tile and Region Processing
# ==========================================
def assign_regions_to_tiles(regions, tile_height, tile_width, rows, cols):
    """Distribute global regions into processing tiles to optimise memory usage."""
    tiles = {(r, c): [] for r in range(rows) for c in range(cols)}
    for region in regions:
        col, row = region['pixel_centroid']
        tile_row = min(int(row / tile_height), rows - 1)
        tile_col = min(int(col / tile_width), cols - 1)
        region['centroid'] = (col - tile_col * tile_width, row - tile_row * tile_height)
        region['tile_row'] = tile_row
        region['tile_col'] = tile_col
        tiles[(tile_row, tile_col)].append(region)
    return tiles

def process_tile_regions(regions, labeled_mask_path, tile_row, tile_col):
    """Extract precise mask and contour data for regions within a specific tile."""
    processed_regions = []
    if not regions or not labeled_mask_path or not os.path.exists(labeled_mask_path):
        return processed_regions
    with rasterio.open(labeled_mask_path) as src:
        for region in regions:
            left, top, width, height = [int(x) for x in region['bbox']]
            left = max(0, left)
            top = max(0, top)
            width = min(width, src.width - left)
            height = min(height, src.height - top)
            
            if width <= 0 or height <= 0:
                region['mask'] = None
                region['contour'] = None
                region['smooth_contour'] = None
                region['axis_ratio'] = 1.0
                processed_regions.append(region)
                continue
                
            window = Window(left, top, width, height)
            mask = src.read(1, window=window)
            mask = (mask == region['label']).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            region['contour'] = max(contours, key=cv2.contourArea) if contours else None
            region['mask'] = mask if np.sum(mask) > 0 else None
            region['smooth_contour'] = smooth_contour(region.get('contour', None))
            region['axis_ratio'] = get_axis_ratio(region['smooth_contour']) if region['smooth_contour'] is not None else 1.0
            processed_regions.append(region)
            
    return processed_regions

def load_regions_from_json(json_path, timestamp, tile_height, tile_width, rows, cols, labeled_mask_path=None, max_workers=8):
    """Load region data from JSON and process masks concurrently."""
    try:
        with open(json_path, 'r') as f:
            regions = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}
   
    timestamp_regions = assign_regions_to_tiles(regions, tile_height, tile_width, rows, cols)
   
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for (tile_row, tile_col), tile_regions in timestamp_regions.items():
            if tile_regions:
                futures.append(
                    executor.submit(
                        process_tile_regions,
                        tile_regions,
                        labeled_mask_path,
                        tile_row,
                        tile_col
                    )
                )
        for future in futures:
            processed_regions = future.result()
            for region in processed_regions:
                tile_row, tile_col = region['tile_row'], region['tile_col']
                timestamp_regions[(tile_row, tile_col)] = [
                    r for r in timestamp_regions[(tile_row, tile_col)]
                    if r['unique_id'] != region['unique_id']
                ] + [region]
   
    return timestamp_regions

# ==========================================
# 3. Trajectory Tracking Logic (Hungarian Algorithm)
# ==========================================
def track_icebergs(regions_per_timestamp_per_tile, timestamps, max_distance, max_missing_frames, output_folder):
    """Establish and maintain iceberg trajectories across sequential frames."""
    tracks = {}
    track_id = 1
    regions_list = []
   
    for t in range(len(timestamps)):
        regions = []
        if t in regions_per_timestamp_per_tile:
            for (tile_row, tile_col), tile_regions in regions_per_timestamp_per_tile[t].items():
                for region in tile_regions:
                    region['file_x'] = tile_col
                    region['file_y'] = tile_row
                    regions.append(region)
        regions_list.append(regions)
   
    if not regions_list:
        print("Error: No regions found for any timestamp.")
        return {}
   
    # Initialise the first frame
    if regions_list[0]:
        for region in regions_list[0]:
            centroid_x, centroid_y = region['centroid']
            tracks[track_id] = {
                'positions': [(timestamps[0], region['file_x'], region['file_y'])],
                'centroids': [(centroid_x, centroid_y)],
                'areas': [region['area']],
                'bboxes': [region['bbox']],
                'unique_ids': [region['unique_id']],
                'last_seen': timestamps[0],
                'location': (region['file_x'], region['file_y']),
                'masks': [region.get('mask', None)],
                'contours': [region.get('contour', None)],
                'smooth_contours': [region.get('smooth_contour', None)],
                'axis_ratios': [region.get('axis_ratio', 1.0)],
                'missing_count': 0,
                'geo_centroids': [region['geo_centroid']],
                'pixel_centroids': [region['pixel_centroid']],
                'cost_data': [None]
            }
            track_id += 1
   
    # Track across subsequent frames
    for t in range(1, len(timestamps)):
        curr_regions = regions_list[t]
        prev_regions = regions_list[t - 1] if t > 0 else []
        total_matched = 0
        total_new = 0
        
        if not curr_regions or not prev_regions:
            print(f"Skipping timestamp {timestamps[t]}: Missing regions or previous regions.")
            continue
            
        curr_regions_by_pos = {}
        for region in curr_regions:
            pos = (region['file_x'], region['file_y'])
            curr_regions_by_pos.setdefault(pos, []).append(region)
        
        prev_tracks = {
            tid: {
                'pos': track['positions'][-1][1:],
                'centroid': track['centroids'][-1],
                'mask': track['masks'][-1],
                'contour': track['smooth_contours'][-1],
                'ratio': track['axis_ratios'][-1],
                'location': track['location'],
                'area': track['areas'][-1],
                'missing_count': track['missing_count'],
                'bbox': track['bboxes'][-1]
            }
            for tid, track in tracks.items()
            if track['missing_count'] <= max_missing_frames
        }
        
        for pos, curr_regions in curr_regions_by_pos.items():
            pos_tracks = {tid: track for tid, track in prev_tracks.items() if track['location'] == pos}
            
            # Case 1: No current regions in this tile, increment missing count for existing tracks
            if not curr_regions:
                for tid in pos_tracks:
                    if tracks[tid]['missing_count'] <= max_missing_frames:
                        tracks[tid]['missing_count'] += 1
                continue
            
            # Case 2: No previous tracks, all current regions are new tracks
            if not pos_tracks:
                for region in curr_regions:
                    tracks[track_id] = {
                        'positions': [(timestamps[t], region['file_x'], region['file_y'])],
                        'centroids': [(region['centroid'][0], region['centroid'][1])],
                        'areas': [region['area']],
                        'bboxes': [region['bbox']],
                        'unique_ids': [region['unique_id']],
                        'last_seen': timestamps[t],
                        'location': (region['file_x'], region['file_y']),
                        'masks': [region.get('mask', None)],
                        'contours': [region.get('contour', None)],
                        'smooth_contours': [region.get('smooth_contour', None)],
                        'axis_ratios': [region.get('axis_ratio', 1.0)],
                        'missing_count': 0,
                        'geo_centroids': [region['geo_centroid']],
                        'pixel_centroids': [region['pixel_centroid']],
                        'cost_data': [None]
                    }
                    track_id += 1
                    total_new += 1
                continue
            
            # Case 3: Calculate cost matrix and apply Hungarian matching
            iou_diffs, axis_diffs, contour_costs, centroid_dists = [], [], [], []
            for curr_region in curr_regions:
                curr_centroid = curr_region['centroid']
                curr_mask = curr_region.get('mask', None)
                curr_contour = curr_region.get('smooth_contour', None)
                curr_ratio = curr_region.get('axis_ratio', 1.0)
                
                for prev_region in prev_regions:
                    if prev_region['file_x'] != curr_region['file_x'] or prev_region['file_y'] != curr_region['file_y']:
                        continue
                        
                    prev_centroid = prev_region['centroid']
                    prev_mask = prev_region.get('mask', None)
                    prev_contour = prev_region.get('smooth_contour', None)
                    prev_ratio = prev_region.get('axis_ratio', 1.0)
                    
                    area_diff = abs(curr_region['area'] - prev_region['area']) / max(prev_region['area'], 1)
                    if area_diff > 0.5: continue
                    
                    if curr_mask is not None and prev_mask is not None:
                        iou = compute_iou(curr_mask, prev_mask, curr_region['bbox'], prev_region['bbox'])
                        area_cost = 1 - iou
                        if area_cost > 0.99: continue
                    else:
                        area_cost = 1.0
                    
                    axis_ratio_diff = abs(curr_ratio - prev_ratio)
                    centroid_dist = distance.euclidean(curr_centroid, prev_centroid)
                    
                    iou_diffs.append(area_cost)
                    axis_diffs.append(axis_ratio_diff)
                    if curr_contour is not None and prev_contour is not None:
                        contour_cost = cv2.matchShapes(curr_contour, prev_contour, cv2.CONTOURS_MATCH_I2, 0)
                        contour_costs.append(contour_cost)
                    centroid_dists.append(centroid_dist)
            
            # Calculate normalisation factors
            norm_iou_factor = max(np.percentile(iou_diffs, 75), 0.05) if iou_diffs else 0.2
            norm_axis_factor = max(np.percentile(axis_diffs, 75), 0.1) if axis_diffs else 1.0
            norm_contour_factor = max(np.percentile(contour_costs, 75), 0.1) if contour_costs else 1.0
            
            cost_matrix = np.zeros((len(curr_regions), len(pos_tracks)))
            track_ids = list(pos_tracks.keys())
            
            for i, region in enumerate(curr_regions):
                curr_centroid = region['centroid']
                curr_mask = region.get('mask', None)
                curr_contour = region.get('smooth_contour', None)
                curr_ratio = region.get('axis_ratio', 1.0)
                curr_area = region['area']
                
                for j, tid in enumerate(track_ids):
                    prev = pos_tracks[tid]
                    prev_area = prev['area']
                    
                    if curr_area <= 0 or prev_area <= 0:
                        cost_matrix[i, j] = 1e6
                        continue
                        
                    centroid_dist = distance.euclidean(curr_centroid, prev['centroid'])
                    if centroid_dist > max_distance:
                        cost_matrix[i, j] = 1e6
                        continue
                        
                    area_diff = abs(curr_area - prev_area) / max(prev_area, 1)
                    if area_diff > 0.5:
                        cost_matrix[i, j] = 1e6
                        continue
                    
                    if curr_mask is not None and prev['mask'] is not None:
                        iou = compute_iou(curr_mask, prev['mask'], region['bbox'], prev['bbox'])
                        area_cost = 1 - iou
                        if area_cost > 0.99:
                            cost_matrix[i, j] = 1e6
                            continue
                    else:
                        area_cost = 1.0
                        
                    axis_ratio_diff = abs(curr_ratio - prev['ratio'])
                    norm_area_cost = area_cost / norm_iou_factor
                    norm_axis_cost = axis_ratio_diff / norm_axis_factor
                    
                    contour_cost = 10.0 if curr_contour is None or prev['contour'] is None else cv2.matchShapes(curr_contour, prev['contour'], cv2.CONTOURS_MATCH_I2, 0)
                    contour_cost = min(contour_cost, 100.0)
                    norm_contour_cost = contour_cost / norm_contour_factor if norm_contour_factor > 1e-6 else 0
                    
                    cost = (0.05 * centroid_dist + 0.30 * norm_area_cost + 0.60 * norm_axis_cost + 0.05 * norm_contour_cost)
                    cost_matrix[i, j] = cost if not np.isnan(cost) and not np.isinf(cost) else 1e6
            
            # Handle ill-conditioned cost matrices
            if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
                for region in curr_regions:
                    tracks[track_id] = {
                        'positions': [(timestamps[t], region['file_x'], region['file_y'])],
                        'centroids': [(region['centroid'][0], region['centroid'][1])],
                        'areas': [region['area']],
                        'bboxes': [region['bbox']],
                        'unique_ids': [region['unique_id']],
                        'last_seen': timestamps[t],
                        'location': (region['file_x'], region['file_y']),
                        'masks': [region.get('mask', None)],
                        'contours': [region.get('contour', None)],
                        'smooth_contours': [region.get('smooth_contour', None)],
                        'axis_ratios': [region.get('axis_ratio', 1.0)],
                        'missing_count': 0,
                        'geo_centroids': [region['geo_centroid']],
                        'pixel_centroids': [region['pixel_centroid']],
                        'cost_data': [None]
                    }
                    track_id += 1
                    total_new += 1
                for tid in pos_tracks:
                    if tracks[tid]['missing_count'] < max_missing_frames:
                        tracks[tid]['missing_count'] += 1
                continue
            
            # Execute Linear Sum Assignment (Hungarian Algorithm)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_regions = set()
            matched_track_ids = set()
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < 1e6:
                    region = curr_regions[i]
                    tid = track_ids[j]
                    
                    # Recalculate costs for logging purposes
                    curr_centroid = region['centroid']
                    curr_mask = region.get('mask', None)
                    curr_contour = region.get('smooth_contour', None)
                    curr_ratio = region.get('axis_ratio', 1.0)
                    prev = pos_tracks[tid]
                    centroid_dist = distance.euclidean(curr_centroid, prev['centroid'])
                    
                    if curr_mask is not None and prev['mask'] is not None:
                        iou = compute_iou(curr_mask, prev['mask'], region['bbox'], prev['bbox'])
                        area_cost = 1 - iou
                    else:
                        area_cost = 1.0
                        
                    axis_ratio_diff = abs(curr_ratio - prev['ratio'])
                    norm_area_cost = area_cost / norm_iou_factor
                    norm_axis_cost = axis_ratio_diff / norm_axis_factor
                    contour_cost = 10.0 if curr_contour is None or prev['contour'] is None else cv2.matchShapes(curr_contour, prev['contour'], cv2.CONTOURS_MATCH_I2, 0)
                    contour_cost = min(contour_cost, 100.0)
                    norm_contour_cost = contour_cost / norm_contour_factor if norm_contour_factor > 1e-6 else 0
                    total_cost = (0.05 * centroid_dist + 0.30 * norm_area_cost + 0.60 * norm_axis_cost + 0.05 * norm_contour_cost)
                    cost_info = [centroid_dist, norm_area_cost, norm_axis_cost, norm_contour_cost, total_cost]
                    
                    # Update Track Dictionary
                    tracks[tid]['positions'].append((timestamps[t], region['file_x'], region['file_y']))
                    tracks[tid]['centroids'].append(region['centroid'])
                    tracks[tid]['areas'].append(region['area'])
                    tracks[tid]['bboxes'].append(region['bbox'])
                    tracks[tid]['unique_ids'].append(region['unique_id'])
                    tracks[tid]['last_seen'] = timestamps[t]
                    tracks[tid]['masks'].append(region.get('mask', None))
                    tracks[tid]['contours'].append(region.get('contour', None))
                    tracks[tid]['smooth_contours'].append(region.get('smooth_contour', None))
                    tracks[tid]['axis_ratios'].append(region.get('axis_ratio', 1.0))
                    tracks[tid]['missing_count'] = 0
                    tracks[tid]['geo_centroids'].append(region['geo_centroid'])
                    tracks[tid]['pixel_centroids'].append(region['pixel_centroid'])
                    tracks[tid]['cost_data'].append(cost_info)
                    
                    matched_regions.add(region['unique_id'])
                    matched_track_ids.add(tid)
                    total_matched += 1
            
            # Unmatched existing tracks increment their missing count
            for tid in pos_tracks:
                if tid not in matched_track_ids and tracks[tid]['missing_count'] <= max_missing_frames:
                    tracks[tid]['missing_count'] += 1
            
            # Unmatched current regions spawn new tracks
            for i, region in enumerate(curr_regions):
                if region['unique_id'] not in matched_regions:
                    tracks[track_id] = {
                        'positions': [(timestamps[t], region['file_x'], region['file_y'])],
                        'centroids': [(region['centroid'][0], region['centroid'][1])],
                        'areas': [region['area']],
                        'bboxes': [region['bbox']],
                        'unique_ids': [region['unique_id']],
                        'last_seen': timestamps[t],
                        'location': (region['file_x'], region['file_y']),
                        'masks': [region.get('mask', None)],
                        'contours': [region.get('contour', None)],
                        'smooth_contours': [region.get('smooth_contour', None)],
                        'axis_ratios': [region.get('axis_ratio', 1.0)],
                        'missing_count': 0,
                        'geo_centroids': [region['geo_centroid']],
                        'pixel_centroids': [region['pixel_centroid']],
                        'cost_data': [None]
                    }
                    track_id += 1
                    total_new += 1
                    
        print(f"Timestamp {timestamps[t]}: Matched {total_matched} regions, created {total_new} new tracks.")
        
    return tracks

def save_tracks_to_csv(all_tracks, output_csv_path, timestamps, num_chunks=4):
    """Save trajectory data to CSV using parallelised chunk processing."""
    track_ids = list(all_tracks.keys())
    if not track_ids:
        print("No tracks available to save.")
        return

    chunk_size = max(1, len(track_ids) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(track_ids)
        if start < len(track_ids):
            chunks.append(track_ids[start:end])
   
    def process_chunk(chunk_ids, chunk_file, all_tracks, timestamps):
        data = []
        for track_id in chunk_ids:
            track = all_tracks[track_id]
            for i, (ts, tile_x, tile_y) in enumerate(track['positions']):
                centroid_x, centroid_y = track['pixel_centroids'][i]
                geo_x, geo_y = track['geo_centroids'][i]
                area = track['areas'][i]
                unique_id = track['unique_ids'][i]
                cost_info = track['cost_data'][i]
                
                cost_data = [None] * 5 if cost_info is None else cost_info
                data.append([
                    track_id, ts, tile_x, tile_y, centroid_x, centroid_y, geo_x, geo_y, area, unique_id,
                    *cost_data
                ])
                
        df = pd.DataFrame(data, columns=[
            'Track_ID', 'Timestamp', 'Tile_X', 'Tile_Y',
            'Centroid_X', 'Centroid_Y', 'Geo_X', 'Geo_Y', 'Area', 'Unique_ID',
            'Centroid_Distance', 'Area_Cost', 'Axis_Cost', 'Contour_Cost', 'Total_Cost'
        ])
        df.to_csv(chunk_file, index=False)
   
    processes = []
    chunk_files = [f"{output_csv_path}.part{i}" for i in range(len(chunks))]
    for i, chunk in enumerate(chunks):
        p = mp.Process(target=process_chunk, args=(chunk, chunk_files[i], all_tracks, timestamps))
        processes.append(p)
        p.start()
   
    for p in processes:
        p.join()
   
    # Merge chunk files
    with open(output_csv_path, 'w') as outfile:
        for i, chunk_file in enumerate(chunk_files):
            if os.path.exists(chunk_file):
                with open(chunk_file, 'r') as infile:
                    lines = infile.readlines()
                    if not lines: 
                        continue
                    if i == 0:
                        outfile.writelines(lines)
                    else:
                        outfile.writelines(lines[1:])
                os.remove(chunk_file)
   
    print(f"Trajectories and assignment costs successfully saved to: {output_csv_path}")

def process_timestamp(args):
    """Wrapper function to handle region loading for a single timestamp."""
    ts_idx, timestamp, base_mask_folder, tile_height, tile_width, rows, cols, max_workers, prefix = args
    json_path = os.path.join(base_mask_folder, f"{prefix}regions_{timestamp}.json")
    labeled_mask_path = os.path.join(base_mask_folder, f"{prefix}labeled_mask_{timestamp}.tif")
    
    if not os.path.exists(json_path):
        print(f"JSON file {json_path} does not exist, skipping.")
        return ts_idx, {}
        
    timestamp_regions = load_regions_from_json(json_path, timestamp, tile_height, tile_width, rows, cols, labeled_mask_path, max_workers)
    
    if not timestamp_regions:
        print(f"No valid regions loaded for {timestamp}, skipping.")
        return ts_idx, {}
        
    print(f"Timestamp {timestamp}: Successfully loaded {sum(len(regions) for regions in timestamp_regions.values())} regions.")
    return ts_idx, timestamp_regions

def process_tile_tracking(base_mask_folder, timestamps, output_csv_name, rows, cols, max_distance, max_missing_frames, max_workers, prefix):
    """Main execution orchestrator for tile-based parallel tracking."""
    output_csv_path = os.path.join(base_mask_folder, output_csv_name)
    regions_per_timestamp_per_tile = {}
    tile_height, tile_width = None, None
   
    for timestamp in timestamps:
        mask_path = os.path.join(base_mask_folder, f"{prefix}labeled_mask_{timestamp}.tif")
        if os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                height, width = src.shape
                tile_height = height // rows
                tile_width = width // cols
            break
            
    if tile_height is None or tile_width is None:
        print(f"Failed to determine tile dimensions in {base_mask_folder}.")
        return {}
   
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_timestamp,
                (ts_idx, timestamp, base_mask_folder, tile_height, tile_width, rows, cols, max_workers, prefix)
            )
            for ts_idx, timestamp in enumerate(timestamps)
        ]
        for future in futures:
            ts_idx, timestamp_regions = future.result()
            if timestamp_regions:
                regions_per_timestamp_per_tile[ts_idx] = timestamp_regions
   
    tracks = track_icebergs(regions_per_timestamp_per_tile, timestamps, max_distance, max_missing_frames, base_mask_folder)
    save_tracks_to_csv(tracks, output_csv_path, timestamps, num_chunks=max(1, max_workers//2))
    
    return tracks

# ==========================================
# 4. Command Line Interface Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-temporal Grounded Iceberg Tracking Algorithm.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing mask and json data.")
    parser.add_argument("--rows", type=int, default=30, help="Number of horizontal tiles for block processing.")
    parser.add_argument("--cols", type=int, default=30, help="Number of vertical tiles for block processing.")
    parser.add_argument("--max_distance", type=float, default=10.0, help="Maximum allowable centroid shift (in pixels) between frames.")
    parser.add_argument("--max_missing_frames", type=int, default=1, help="Maximum number of consecutive missing frames allowed before track termination.")
    args = parser.parse_args()

    # Define system capabilities
    max_workers = min(mp.cpu_count(), 12)
    root_dir = args.input_dir

    if not os.path.exists(root_dir):
        print(f"Error: Input directory {root_dir} does not exist.")
        return

    # Determine processing priority (coastline takes precedence over raw)
    has_coastline = any(f.startswith('coastline_regions_') for f in os.listdir(root_dir))
    prefix = "coastline_" if has_coastline else "raw_"
    
    print(f"\n{'='*50}")
    print(f"Processing Directory: {root_dir}")
    print(f"Detected processing mode: {prefix.strip('_')}")
    print(f"{'='*50}")
    
    output_csv_name = f"Grounded_iceberg_tracks_{prefix.strip('_')}.csv"
    output_csv_path = os.path.join(root_dir, output_csv_name)
    
    if os.path.exists(output_csv_path):
        print(f"\n[SKIP] Output file '{output_csv_name}' already exists.")
        return

    # Parse available timestamps from JSON files
    timestamps = []
    try:
        for f in os.listdir(root_dir):
            if f.startswith(f'{prefix}regions_') and f.endswith('.json'):
                # Extract the 15-character timestamp part
                timestamp_part = f.split('_')[-1].split('.')[0]
                if len(timestamp_part) == 15:
                    timestamps.append(timestamp_part)
        
        timestamps = sorted(list(set(timestamps)))
        
        if not timestamps:
            print(f"Warning: No valid sequential {prefix.strip('_')} JSON data discovered in the directory. Bypassing.")
            return
            
        print(f"Identified {len(timestamps)} chronological timestamps. Commencing trajectory analysis...")
        
        start_time = time.time()
        process_tile_tracking(
            base_mask_folder=root_dir,
            timestamps=timestamps,
            output_csv_name=output_csv_name,
            rows=args.rows,
            cols=args.cols,
            max_distance=args.max_distance,
            max_missing_frames=args.max_missing_frames,
            max_workers=max_workers,
            prefix=prefix
        )
        print(f"Sequence successfully analysed and compiled in {time.time() - start_time:.2f} seconds.")
        
    except Exception as e:
        print(f"CRITICAL ERROR processing sequence: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
