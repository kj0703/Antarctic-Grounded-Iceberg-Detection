# Antarctic-Grounded-Iceberg-Detection

Source code for the ESSD paper: *'A circum-Antarctic grounded iceberg dataset derived from Sentinel-1'*.

## Overview
This repository contains a comprehensive, automated pipeline for the detection, tracking, and geospatial analysis of grounded icebergs using multi-temporal synthetic aperture radar (SAR) imagery. Designed for high-performance computing (HPC) environments, the workflow integrates deep learning segmentation, Hungarian tracking algorithms, physical environmental quality control, and robust spatial topology classification.

## Pipeline Architecture
The workflow is divided into five sequential Python modules. It utilises a flattened directory architecture for streamlined I/O operations and automatic schema detection for robust data handling.

### 1. Deep Learning Segmentation (`detect.py`)
Utilises a ResUNet architecture to predict binary iceberg masks from Sentinel-1 SAR imagery.
* **Input**: Calibrated SAR imagery (TIF).
* **Output**: Binary prediction masks.

### 2. Region Extraction & Coastline Masking (`extract_icebergs.py`)
Extracts connected components from the binary masks. It includes an optional spatial intersection filter using the high-resolution SCAR Antarctic Digital Database (ADD) to eliminate terrestrial and ice-shelf false positives. Outputs are automatically prefixed (`coastline_` or `raw_`) for data traceability.
* **Input**: Binary prediction masks.
* **Output**: Labelled TIF masks and JSON region properties.

### 3. Multi-temporal Trajectory Analysis (`identify_stationaries.py`)
Implements a bipartite matching sequence (Hungarian algorithm) to identify stationary iceberg trajectories across multiple temporal frames. It automatically detects and prioritises coastline-filtered data.
* **Input**: Labelled masks and JSON region files.
* **Output**: Tabular trajectory dataset (`.csv`).

### 4. Physical Area Correction and Environmental QC (`physical_filter.py`)
Performs physics-based post-processing. It applies latitude-dependent pixel scaling for accurate area calculation (EPSG:3031) and integrates external datasets (IBCSO bathymetry, OSI-SAF Sea Ice Concentration) to filter out deep-water anomalies and sea-ice false positives. Thresholds are customisable for both IW and EW swath modes.
* **Input**: Trajectory CSV.
* **Output**: Refined trajectory dataset (`_Final.csv`).

### 5. Geospatial Synthesis & Deduplication (`export_geopackage.py`)
The final data packaging module. It vectorises the representative polygon for each iceberg from its temporal midpoint, resolves spatial overlaps across adjacent orbital passes via R-Tree deduplication, and conducts topological intersection analysis against reference landfast ice geometries.
* **Input**: Refined trajectory CSV and corresponding TIF masks.
* **Output**: Fully attributed GeoPackage datasets (`.gpkg`), structured in both flat and architectural layered formats (Inside, Partial, Outside fast ice).

---

## Usage: Automated Pipeline Execution
Below is a standard execution sequence for a typical orbital dataset. *Note: The physical filter parameters in Step 4 are currently calibrated for Extra Wide (EW) swath mode.*

```bash
# Step 1: Deep Learning Prediction
python detect.py \
  --input_dir "/path/to/TIF" \
  --output_dir "/path/to/Mask" \
  --model_path "/path/to/resunet_v1.0.0.pth" \
  --patch_size 250 \
  --batch_size 35

# Step 2: Extraction and Coastline Masking
python extract_icebergs.py \
  --input_dir "/path/to/Mask" \
  --output_dir "/path/to/Extracted" \
  --min_area 10 \
  --block_rows 1 \
  --block_cols 1 \
  --coastline_gpkg "/path/to/SCAR_ADD_coastline.gpkg"

# Step 3: Trajectory Identification
python identify_stationaries.py \
  --input_dir "/path/to/Extracted" \
  --rows 30 \
  --cols 30 \
  --max_distance 10.0 \
  --max_missing_frames 1

# Step 4: Physical and Environmental Filtering
# Example for Extra Wide (EW) mode:
python physical_filter.py \
  --input_dir "/path/to/Extracted" \
  --bathy_file "/path/to/IBCSO_bed.tif" \
  --sic_dir "/path/to/SIC" \
  --min_pixels 10 \
  --cond1_area 625 \
  --cond2_area 63

# Example for Interferometric Wide (IW) mode:
# python physical_filter.py \
#   --input_dir "/path/to/Extracted" \
#   --bathy_file "/path/to/IBCSO_bed.tif" \
#   --sic_dir "/path/to/SIC" \
#   --min_pixels 40 \
#   --cond1_area 2500 \
#   --cond2_area 250

# Step 5: Geospatial Export and Classification
python export_geopackage.py \
  --input_csv "/path/to/Extracted/Grounded_iceberg_tracks_coastline_Final.csv" \
  --mask_dir "/path/to/Extracted" \
  --fast_ice_gpkg "/path/to/fast_ice_reference.gpkg" \
  --output_dir "/path/to/Final_Product" \
  --iou_threshold 0.5 \
  --max_workers 6
```
---

## Data Coordinate Reference System (CRS)
The standard projection utilised throughout this pipeline is **EPSG:3031** (WGS 84 / Antarctic Polar Stereographic).

---
## Declaration of Generative AI in Scientific Coding
*During the preparation of this repository, generative AI tools (Gemini3) were utilised to assist in code refactoring, structural optimisation, and the drafting of documentation. All AI-generated code was thoroughly reviewed, rigorously tested in the NCI supercomputing environment, and validated by the primary author to ensure academic integrity and algorithmic accuracy.*
