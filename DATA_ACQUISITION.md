# Ancillary Data Acquisition Guide

> **Note for Demo Users**
> If you are only evaluating the pipeline using the provided `demo_data.zip`, you do **not** need to download these datasets manually. Miniature, pre-cropped versions of all required ancillary data (coastline, bathymetry, sea ice concentration, and fast ice) are already included within the demo archive. 
> 
> The instructions below are intended strictly for users preparing to process their own full-scale orbital datasets.

---

To successfully run the environmental filtering and spatial topological classification modules of this pipeline on new data, you must download the following ancillary datasets. Ensure they are stored locally and properly referenced in the command-line arguments.

## 1. SCAR Antarctic Digital Database (ADD) Coastline
Used in `src/extract_icebergs.py` to mask out terrestrial landmasses, ice shelves, and ice tongues.
* **Source**: [SCAR Antarctic Digital Database](https://www.add.scar.org/)
* **Required Product**: High-resolution coastline polygon (Version 7.11 or newer).
* **Format**: Download as Shapefile and convert to GeoPackage (`.gpkg`), or download directly as `.gpkg` if available.
* **Target Layer Name**: The script expects the layer to be named `add_coastline_high_res_polygon_v7_11`.

## 2. IBCSO Bathymetry
Used in `src/physical_filter.py` to eliminate false positives occurring in deep ocean waters.
* **Source**: [International Bathymetric Chart of the Southern Ocean (IBCSO)](https://ibcso.org/) / PANGAEA.
* **Required Product**: IBCSO v2 Bed Elevation dataset.
* **Format**: GeoTIFF (`.tif`).
* **Note**: The pipeline handles dynamic reprojection to EPSG:3031 internally via `Rasterio.WarpedVRT`; you do not need to manually reproject the source file.

## 3. OSI-SAF Sea Ice Concentration (SIC)
Used in `src/physical_filter.py` to filter out anomalies caused by dense sea ice and ice mélange.
* **Source**: EUMETSAT OSI-SAF Data Centre.
* **Required Product**: Global Sea Ice Concentration (SSMIS / AMSR2) - Antarctic Polar Stereographic projection.
* **Format**: NetCDF (`.nc`).
* **File Naming**: Ensure files maintain the naming convention containing the date string (e.g., `*YYYYMMDD*.nc`) so the pipeline can match them temporally with the SAR imagery.

## 4. Landfast Ice Reference Dataset
Used in `src/export_geopackage.py` for spatial intersection to classify icebergs as *Inside*, *Partial*, or *Outside* the fast ice extent.
* **Source**: Provided by Alex Fraser's research group.
* **Format**: GeoPackage (`.gpkg`).
* **Requirement**: The script evaluates topological overlaps against geometries designated with `class_id == 1`. Ensure your vector data aligns with this schema.
