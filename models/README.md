# Models Directory

This directory is designated for storing pre-trained deep learning weights used by the pipeline.

## Required Weight File
The core segmentation module (`detect.py`) requires the following weight file to function:
- **File Name**: `resunet_v1.0.pth`
- **Size**: ~300 MB

## Download Instructions
Due to GitHub's file size limitations, the weights are hosted in the repository's **Releases** section rather than being tracked directly by Git.

1.  **Download**: Obtain the weights from the [Initial release of pre-trained weights](https://github.com/kj0703/Antarctic-Grounded-Iceberg-Detection/releases/tag/v1.0.0).
2.  **Placement**: After downloading, place the `.pth` file directly into this `models/` folder.
3.  **Verification**: Ensure the final path is `models/resunet_v1.0.pth` before running `detect.py`.
