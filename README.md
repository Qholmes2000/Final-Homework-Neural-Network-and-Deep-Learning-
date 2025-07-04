# README

## **1.** ​Neural Radiance Fields for Object Reconstruction and Novel View Synthesis​
## (1) Overview
This repository demonstrates how to train a Neural Radiance Field (NeRF) model using the Nerfacto/TensoRF/Vanilla-NeRF(The Original NERF) method from Nerfstudio. The repository contains three methods configuration file, which provided (`config.yml`) contains all the necessary settings to replicate our training process.

The first experiment reconstructs 3D scenes using three methods from Nerfstudio. Key features were extracted and matched using COLMAP SfM, establishing accurate camera poses through sparse reconstruction. The pipeline transformed COLMAP's sparse outputs into standardized transform files compatible with NeRF training. Models were trained for ​30,000 iterations​ with a ​ray batch size of 4,096, using the ​Adam optimizer​ with customized learning rates.

## (2) Key Features
- **Method**: Nerfacto (an optimized NeRF implementation)
- **Training Device**: NVIDIA CUDA GPU
- **Data Preprocessing**: Auto-scaled poses with orientation correction
- **Efficiency Optimization**: Multi-resolution hash encoding and proposal networks

## (3) Requirements
1. Nerfstudio installed (official installation guide: [nerf.studio](https://docs.nerf.studio/))
2. NVIDIA GPU with CUDA support
3. Python 3.8+
4. PyTorch 2.0+
5. Colmap 3.8+
6. Tiny-cuda-nn 1.7

## (4) Data Processing Pipeline with Nerfstudio & COLMAP

This section explains how to process raw images into a NeRF-ready dataset using Nerfstudio and COLMAP. This pipeline converts unstructured photos into the processed format required for NeRF training.

### Processing Pipeline Steps

1. **Capture Raw Images**
2. **Run COLMAP Reconstruction**
3. **Convert to Nerfstudio Format**
4. **Post-process and Validate**

### Step-by-Step Processing Guide

#### 1. Prepare Raw Images
- Place images in a directory (in our reposity, e.g.`~/images`)
- Requirements:
  - Minimum 20 images (100+ recommended)
  - Consistent lighting conditions
  - Overlapping coverage from different angles
  - Avoid motion blur and reflections

#### 2. Install Required Tools
```bash
# Install COLMAP and Nerfstudio
pip install nerfstudio
conda install -c conda-forge colmap
```

#### 3. Process with Nerfstudio Pipeline
```bash
# Basic processing command
ns-process-data images \
    --data ~/images \
    --output-dir ~/processed_data
```

### Advanced Processing Options

#### Handling Different Camera Types
```bash
# Fisheye Lens Processing
ns-process-data images \
    --camera-model OPENCV_FISHEYE \
    --fisheye-crop-radius 600 \
    ...

# 360° Camera Processing
ns-process-data images \
    --camera-model EQUIRECTANGULAR \
    ...
```

#### Parameter Tuning for Challenging Scenes
```bash
# For low-texture scenes
ns-process-data images \
    --feature-type disk \
    --matcher-type brute-force \
    ...

# For large-scale scenes
ns-process-data images \
    --sfm-tool colmap \
    --use-gpu 1 \
    --max-num_features 8192 \
    ...
```

### Output Directory Structure
After processing, you'll get:
```
processed_data/
├── images/                 # Processed images (scaled, undistorted)
├── masks/                  # Automatic background masks (if available)
├── transforms.json         # Camera parameters in NeRF format
├── colmap/
│   ├── sparse/             # COLMAP sparse reconstruction
│   ├── dense/              # Depth maps (if enabled)
│   └── database.sqlite     # Feature matching database
```

## (5) Validation Commands

#### Check Camera Alignment
```bash
ns-viewer --load-config ~/processed_data/transforms.json
```

#### Inspect Point Cloud
```bash
meshlab ~/processed_data/colmap/sparse_pc.ply
```
## (6) Dataset Preparation
Place processed dataset at:  
`/processed_data/train_processed_data`(in our repository)

## (7) Training Instructions

### Nerfacto Model
#### 1. Basic Training Command
```bash
ns-train nerfacto \
    --data /processed_data/train_processed_data \
    --experiment-name train_processed_data \
    --output-dir New_results/nerfacto \
    --max-num-iterations 30000
```

This is the basic command to train nerfacto. It simply defines:
*   Location of the training dataset
*   Output directory for results
*   Training iterations (`max_steps`) set to 30,000 (for initial training and debugging)

To configure more complex training hyperparameters, create a `.yml` configuration file and run training using the command below.

#### 2. Use the Configuration File
For exact replication of our training configuration:
```bash
ns-train nerfacto --config /path/to/config.yml
```

#### 3. Resume Training
To continue training from a checkpoint:
```bash
ns-train nerfacto --load-dir /path/to/checkpoints
```

#### Nerfacto Configuration Highlights
##### Optimization Parameters
- **Learning Rate**: 0.01 (fields) with cosine decay to 0.0001
- **Batch Size**: 4096 rays/batch
- **Mixed Precision**: Enabled for faster training
- **Proposal Networks**: 2-level hierarchical sampling

##### Model Architecture
- **Multi-resolution Hash Encoding**: 16 levels
- **Hash Map Size**: 2^19
- **Hidden Dimensions**: 64
- **Appearance Embedding**: 32-dimensional

##### Evaluation
- Automatic evaluation every 500 steps
- Full evaluation of all images every 25,000 steps
- Test PSNR tracking for quality monitoring


### TensoRF Model
#### 1. Basic Training Command
```bash
ns-train tensorf \
    --data /processed_data/train_processed_data nerfstudio-data\
    --output-dir New_results/tensorf \
    --max-num-iterations 30000
```

This is the basic command to train TensoRF. It simply defines:
*   Location of the training dataset
*   Output directory for results
*   Training iterations (`max_steps`) set to 30,000 (for initial training and debugging)

To configure more complex training hyperparameters, create a `.yml` configuration file and run training using the command below.

#### 2. Use the Configuration File
For exact replication of our training configuration:
```bash
ns-train tensorf --config /path/to/config.yml
```

#### 3. Resume Training
To continue training from a checkpoint:
```bash
ns-train tensorf --load-dir /path/to/checkpoints
```

### Vanilla-NeRF Model(Original NeRF)
#### 1. Basic Training Command
```bash
ns-train vanilla-nerf \
    --data /processed_data/train_processed_data nerfstudio-data\
    --output-dir New_results/vanilla-nerf \
    --max-num-iterations 30000
```

This is the basic command to train Vanilla-NeRF. It simply defines:
*   Location of the training dataset
*   Output directory for results
*   Training iterations (`max_steps`) set to 30,000 (for initial training and debugging)

To configure more complex training hyperparameters, create a `.yml` configuration file and run training using the command below.

#### 2. Use the Configuration File
For exact replication of our training configuration:
```bash
ns-train vanilla-nerf --config /path/to/config.yml
```

#### 3. Resume Training
To continue training from a checkpoint:
```bash
ns-train vanilla-nerf  --load-dir /path/to/checkpoints
```

## Tips for Best Results
1. Ensure sufficient lighting and texture in input images
2. Provide accurate camera poses
3. Verify dataset contains 90% training and 10% validation images
4. Use the cosine learning rate decay schedule
5. Maintain consistent scale across all scene elements

For additional customization, modify the provided `config.yml` file according to Nerfstudio's configuration documentation.


## **2.** Object Reconstruction and Novel View Synthesis via 3D Gaussian Splatting

### (1) Overview  
This experiment reconstructs 3D scenes using Gaussian Splatting. Features were extracted, matched, and sparse reconstruction performed on the image dataset using COLMAP to obtain camera poses. Official scripts converted COLMAP outputs into the format required for training 3D Gaussian models. Models were trained for **30,000 iterations** with a **batch size of 1**, **Adam optimizer** (learning rate: 0.01), and an **L1 + SSIM loss function**. Training progress (loss curves, PSNR) was monitored via Tensorboard. Post-training, novel trajectory videos rendering object rotations were generated, and quantitative evaluation was performed on a held-out **20% test set** using PSNR, SSIM, and LPIPS metrics, confirming the effectiveness of 3D Gaussian Splatting for 3D reconstruction.

### (2) Environment Setup  
- **Python**: 3.10  
- **PyTorch Installation**:  
  ```bash
  conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
  ```

### (3) Data Preparation  
#### **a. Image Capture**  
- Capture >100 images of the object from multiple viewpoints (e.g., `r_0.png`, `r_1.png`, ..., `r_99.png`).  
- Place them in: `gaussiansplatting/data/custom/train/`  

#### **b. COLMAP Sparse Reconstruction - Feature Extraction**  
```bash
colmap feature_extractor  
```
*Output*: `custom.db`  

#### **c. COLMAP Sparse Reconstruction - Feature Matching**  
```bash
colmap exhaustive_matcher  
```

#### **d. COLMAP Sparse Reconstruction - Sparse Reconstruction**  
```bash
colmap mapper  
```
*Output* (in `colmap/custom_sparse/0/`):  
- Camera parameters: `cameras.txt`  
- Camera poses: `images.txt`  
- Sparse point cloud: `points3D.txt`  

#### **e. Convert to Training Format**  
```bash
python convert.py  
```
*Output*:  
- Training/test image symlinks in `data/custom/processed/train/` & `test/`  
- Camera config files: `transforms_train.json` & `transforms_test.json`  

### (4) Model Training  
- **Iterations**: 30,000; Gaussian points densified every 3,000 iterations.  
- **Optimizer**: Adam  
  - Learning rates: position (0.00016), opacity (0.05), scale (0.005), rotation (0.001).  
- **Loss**: L1 (weight: 0.8) + SSIM (weight: 0.2).  
- **Spherical Harmonics**: `sh_degree=3` for color encoding.  
- **Checkpoints**: Models evaluated/saved at iterations 1k, 5k, 10k, 20k, 30k.  

### (5) Novel View Video Rendering  
```bash
python render.py  
```

### (6) Quantitative Test Set Evaluation  
```bash
python full_eval.py  
```
*Metrics*: PSNR, SSIM, LPIPS. Results saved to `eval_results.json`.  

### (7) Results  
**Training monitoring**:  
- Tensorboard logs: `output/custom_model/tensorboard/` (loss curves, PSNR).  

**Quantitative comparison**:  
3D Gaussian Splatting significantly outperforms NeRF, TensoRF, and Nerfacto:  
| Method     | SSIM ↑  | PSNR ↑ | LPIPS ↓ |  
|------------|---------|--------|---------|  
| NeRF       | 0.89    | 26.04  | 0.090   |  
| TensoRF    | 0.90    | 26.81  | 0.090   |  
| Nerfacto   | 0.89    | 25.34  | 0.070   |  
| **3D-GS**  | **0.97**| **35.90**| **0.010** |  
