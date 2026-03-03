# Multi-Task Hand Gesture Recognition

## Project Overview
This project implements a multi-task deep learning model to simultaneously perform hand gesture classification, bounding box regression, and semantic segmentation using RGB-D data.

Project Structure
```text
project_25055639_CHENG/
├── data/                   # Directory for dataset (training and test sets)
├── weights/                # Directory for saving trained model weights
├── results/                # Directory for saving evaluation results and visualizations
├── requirements.txt        # Python dependencies
└── src/                    # Source code
    ├── dataloader.py       # Custom Dataset class, data loading, and Albumentations augmentation pipeline
    ├── model.py            # MultiTaskHandNet architecture (Encoder, Classification, BBox, Segmentation heads)
    ├── train.py            # Main training loop with validation, loss calculation, and early stopping
    ├── evaluate.py         # Evaluation script (Accuracy, Box IoU, Mask IoU, and Confusion Matrix generation)
    ├── visualise.py        # Qualitative visualization of model predictions (best vs worst samples)
    ├── visualise_augmentation.py # Visualization of the data augmentation effects
    ├── utils.py            # Utility functions for calculating metrics (e.g., Intersection over Union)
    └── extract_data.py     # Script to process and organize raw data into structured train/test folders
```

## Setup Instructions

### 1. Environment Setup
It is highly recommended to use a virtual environment (like Anaconda or `venv`) to manage dependencies.
```bash
conda create -n env python=3.10
conda activate env
```

### 2. Install Dependencies
Install all the required Python libraries using the provided `requirements.txt` file.
```bash
pip install -r requirements.txt
```

## How to Use

### 1. Data Preparation
Before training, you need to ensure the data is extracted and structured properly. After modifying and confirming the paths in `extract_data.py` are correct, run:
```bash
python src/extract_data.py
```

### 2. Check Data Augmentation
To verify that the Albumentations data augmentation pipeline is correctly applied to RGB images, Depth maps, and Segmentation Masks, run:
```bash
python src/visualise_augmentation.py
```
*Output image will be saved in the `results/` directory.*

### 3. Training the Model
To start training the multi-task network, execute the training script. run:
```bash
python src/train.py
```
*The best model weights based on validation loss will be automatically saved to `weights/best_model.pth`.*

### 4. Evaluation
To evaluate the trained model on the Train, Validation, and Test sets. This script will output quantitative metrics and generate confusion matrices for classification performance.
```bash
python src/evaluate.py
```
*Confusion matrices will be saved as PNG files in the `results/` directory.*

### 5. Visualizing Predictions
To qualitatively assess the model's performance, run the visualization script. This will process a batch from the test set, score the predictions, and generate a plot comparing Ground Truth vs Predictions for the "Best" and "Worst" performing samples.
```bash
python src/visualise.py
```
*The resulting visualization will be saved as `results/visualisation_best_worst.png`.*

