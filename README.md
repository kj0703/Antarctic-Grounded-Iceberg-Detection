# Antarctic-Grounded-Iceberg-Detection

This repository provides a comprehensive, automated pipeline for the detection, tracking, and geospatial analysis of grounded icebergs using multi-temporal synthetic aperture radar (SAR) imagery. Designed for high-performance computing (HPC) environments, the workflow integrates deep learning segmentation, Hungarian tracking algorithms, physical environmental quality control, and robust spatial topology classification.

Source code for the ESSD paper:

**'Grounded Icebergs around Antarctica: A High-Resolution Dataset Derived from Deep Learning and Sentinel-1 Synthetic Aperture Radar'**

---

## 1) Preparation

The pipeline relies heavily on deep learning (PyTorch) and geospatial libraries (GDAL, GeoPandas, Rasterio). It is highly recommended to run the code in a virtual environment.

The simplest way to configure the exact dependencies used on the NCI Gadi supercomputing environment is via the provided `environment.yml` file:

```bash
# create new conda environment from the yml file
conda env create -f environment.yml

# activate environment
conda activate iceberg_detection
```

(Alternatively, a `requirements.txt` is provided for standard pip installations).

---

## 2) Installation

You can set up this library directly on your local machine or HPC environment after cloning the repository.

```bash
# clone the repository
git clone https://github.com/[your-username]/Antarctic-Grounded-Iceberg-Detection.git

# change into the main directory
cd Antarctic-Grounded-Iceberg-Detection
```

---

## 3) Usage: Pipeline Execution

The workflow is divided into five sequential Python modules. It utilises a flattened directory architecture for streamlined I/O operations and automatic schema detection for robust data handling.

Below is a standard execution sequence for a typical orbital dataset.

---

### 3.1) Deep Learning Segmentation (`detect.py`)

Utilises a ResUNet architecture to predict binary iceberg masks from calibrated Sentinel-1 SAR imagery.

```bash
python detect.py \
  --input_dir "/path/to/TIF" \
  --output_dir "/path/to/Mask" \
  --model_path "/path/to/resunet_v1.0.0.pth" \
  --patch_size 250 \
  --batch_size 35
```

---

### 3.2) Region Extraction & Coastline Masking (`extract_icebergs.py`)

Extracts connected components from the binary masks. It includes an optional spatial intersection filter using the high-resolution SCAR Antarctic Digital Database (ADD) to eliminate terrestrial and ice-shelf false positives. Outputs are automatically prefixed for data traceability.

```bash
python extract_icebergs.py \
  --input_dir "/path/to/Mask" \
  --output_dir "/path/to/Extracted" \
  --min_area 10 \
  --block_rows 1 \
  --block_cols 1 \
  --coastline_gpkg "/path/to/SCAR_ADD_coastline.gpkg"
```

---

### 3.3) Multi-temporal Trajectory Analysis (`identify_stationaries.py`)

Implements a bipartite matching sequence (Hungarian algorithm) to identify stationary iceberg trajectories across multiple temporal frames. It automatically detects and prioritises coastline-filtered data.

```bash
python identify_stationaries.py \
  --input_dir "/path/to/Extracted" \
  --rows 30 \
  --cols 30 \
  --max_distance 10.0 \
  --max_missing_frames 1
```

---

### 3.4) Physical Area Correction and Environmental QC (`physical_filter.py`)

Performs physics-based post-processing. It applies latitude-dependent pixel scaling for accurate area calculation and integrates external datasets (IBCSO bathymetry, OSI-SAF Sea Ice Concentration) to filter out deep-water anomalies and sea-ice false positives.

**Example for Extra Wide (EW) swath mode:**

```bash
python physical_filter.py \
  --input_dir "/path/to/Extracted" \
  --bathy_file "/path/to/IBCSO_bed.tif" \
  --sic_dir "/path/to/SIC" \
  --min_pixels 10 \
  --cond1_area 625 \
  --cond2_area 63
```

**Example for Interferometric Wide (IW) swath mode:**

```bash
# python physical_filter.py \
#   --input_dir "/path/to/Extracted" \
#   --bathy_file "/path/to/IBCSO_bed.tif" \
#   --sic_dir "/path/to/SIC" \
#   --min_pixels 40 \
#   --cond1_area 2500 \
#   --cond2_area 250
```

---

### 3.5) Geospatial Synthesis & Deduplication (`export_geopackage.py`)

The final data packaging module. It vectorises the representative polygon for each iceberg from its temporal midpoint, resolves spatial overlaps across adjacent orbital passes via R-Tree deduplication, and conducts topological intersection analysis against reference landfast ice geometries.

```bash
python export_geopackage.py \
  --input_csv "/path/to/Extracted/Grounded_iceberg_tracks_coastline_Final.csv" \
  --mask_dir "/path/to/Extracted" \
  --fast_ice_gpkg "/path/to/fast_ice_reference.gpkg" \
  --output_dir "/path/to/Final_Product" \
  --iou_threshold 0.5 \
  --max_workers 6
```

---

## 4) Data Coordinate Reference System (CRS)

The standard projection utilised throughout this pipeline is **EPSG:3031 (WGS 84 / Antarctic Polar Stereographic)**.

---

## 5) Declaration of Generative AI in Scientific Coding

During the preparation of this repository, generative AI tools (Gemini3) were utilised to assist in code refactoring, structural optimisation, and the drafting of documentation. All AI-generated code was thoroughly reviewed, rigorously tested in the NCI supercomputing environment, and validated by the primary author to ensure academic integrity and algorithmic accuracy.
````
