# CoP-Balance-Feature-Extraction-Classification

This repository contains MATLAB and Python code for feature extraction and machine learning classification of center of pressure (CoP) data recorded during standing balance tasks using the BTrackS Balance Tracking System.

## Project overview

This project focuses on:
1. extracting time-domain and time-frequency features from CoP signals
2. building datasets for classification
3. classifying balance-task conditions using machine learning models

The work is related to standing balance experiments in older adults across four conditions:
- firm surface, eyes open
- firm surface, eyes closed
- foam surface, eyes open
- foam surface, eyes closed

## Repository structure

- `matlab/`
  - `extract_time_features.m`: extracts time-domain CoP features and builds the dataset
  - `extract_cwt_features.m`: extracts time-frequency features using continuous wavelet transform (CWT)

- `python/`
  - `cop_classification.py`: machine learning classification pipeline
  - `cop_classification.ipynb`: notebook version of the classification workflow

- `data/`
  - documentation about expected input data format

## Feature extraction

### 1. Time-domain features
The MATLAB script `extract_time_features.m` extracts CoP time-domain features such as:
- distance
- area
- angle
- speed
- acceleration

### 2. Time-frequency features
The MATLAB script `extract_cwt_features.m` extracts CWT-based time-frequency features from CoP signals.

## Classification

The Python classification pipeline includes:
- loading MATLAB `.mat` datasets
- feature preparation
- train/test split
- classification using:
  - KNN
  - LDA
  - Naive Bayes
  - Random Forest
- performance evaluation

## Requirements

Python packages may include:
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- statsmodels
- h5py
- pywt

Optional packages used in some sections:
- shap
- lime
- keras
- imbalanced-learn

## Data availability

Raw data are not included in this repository unless explicitly permitted.  
Users should prepare input data in MATLAB `.mat` format consistent with the scripts.

## Author

Negar Rahimi