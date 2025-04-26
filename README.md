# Biomass Moisture Prediction from NIR Spectra

## Project Overview
In this project, I developed machine learning models to predict wood chip moisture content using Near-Infrared (NIR) spectroscopy data. My analysis focused on 1,037 NIR spectral readings collected from 125 wood samples, addressing key challenges in high-dimensional spectral data processing and limited sample sizes.

## Technical Implementation

### Data Preprocessing
The pipeline incorporated advanced spectral preprocessing techniques:
- **Savitzky-Golay** filtering for noise reduction and derivative calculations
- **Standard Normal Variate (SNV)** transformation for scattering correction
- **First and second derivatives** for baseline removal and peak resolution

### Machine Learning Models
Three regression approaches were evaluated:
- **Partial Least Squares (PLS)**: Linear dimensionality reduction
- **Support Vector Regression (SVR)**: Non-linear kernel methods
- **Artificial Neural Networks (ANN)**: Deep learning approach

### Outlier Detection Methodology
Implemented PCA-based approach using:
- **Q-residuals** for model deviation
- **Hotelling's T²** for subspace variation
Identified and handled two outlier types:
- **Single-spectrum measurement errors** (removed)
- **Valid anomalous samples** (retained)


## Key Results

## Model Performance Comparison

| Preprocessing          | Model | RMSECV  | R²CV    | RMSE_test | R²_test  |
|------------------------|-------|---------|---------|-----------|----------|
| Raw                    | PLS   | 2.761488 | 0.970050 | 2.659668  | 0.965082 |
| Raw                    | SVR   | 3.329321 | 0.975732 | 3.191873  | 0.949710 |
| Raw                    | ANN   | 5.621636 | 0.844480 | 5.794510  | 0.834260 |
| SG                     | PLS   | 2.751254 | 0.968620 | 2.595997  | 0.966734 |
| SG                     | SVR   | 3.333581 | 0.975596 | 3.195864  | 0.949584 |
| SG                     | ANN   | 5.337949 | 0.863252 | 5.518709  | 0.849662 |
| SNV                    | PLS   | 2.461186 | 0.978307 | 2.128160  | 0.977644 |
| SNV                    | SVR   | 2.270929 | 0.995564 | 2.147867  | 0.977228 |
| SNV                    | ANN   | 3.761335 | 0.929477 | 3.995676  | 0.921191 |
| SG+SNV                 | PLS   | 2.426457 | 0.976278 | 2.204212  | 0.976017 |
| SG+SNV                 | SVR   | 2.291508 | 0.994768 | 2.178501  | 0.976573 |
| SG+SNV                 | ANN   | 3.879931 | 0.937328 | 3.951739  | 0.922915 |
| SG+1stDeriv            | PLS   | 2.845467 | 0.969234 | 2.818440  | 0.960789 |
| SG+1stDeriv            | SVR   | 2.521938 | 0.999748 | 2.563412  | 0.967564 |
| SG+1stDeriv            | ANN   | 4.747749 | 0.892773 | 4.591143  | 0.895952 |
| SG+2ndDeriv            | PLS   | 2.903143 | 0.972759 | 3.056202  | 0.953894 |
| SG+2ndDeriv            | SVR   | 9.067086 | 0.999952 | 8.923595  | 0.606927 |
| SG+2ndDeriv            | ANN   | 5.305074 | 0.851588 | 5.411864  | 0.855427 |
| SG+SNV+1stDeriv        | PLS   | 2.683745 | 0.976068 | 2.575684  | 0.967253 |
| SG+SNV+1stDeriv        | SVR   | 4.895895 | 0.999953 | 4.397647  | 0.904537 |
| SG+SNV+1stDeriv        | ANN   | 3.282732 | 0.953491 | 3.386742  | 0.943382 |
| SG+SNV+2ndDeriv        | PLS   | 2.700008 | 0.976880 | 2.675703  | 0.964660 |
| SG+SNV+2ndDeriv        | SVR   | 11.382660| 0.999953 | 11.150904 | 0.386219 |
| SG+SNV+2ndDeriv        | ANN   | 3.098031 | 0.955420 | 3.217117  | 0.948911 |


### Optimal Model Performance
- **PLS** with SNV preprocessing achieved best results:
  - R² = 0.977, RMSE = 2.13
- **SVR** with SNV performed comparably (R² = 0.977) for non-linear relationships
- **ANN** showed improvement with combined preprocessing but required more data

### Critical Findings
- **SNV normalization** provided the most significant improvement across all models
- **Second derivatives** caused overfitting, especially in SVR (test R² dropped to 0.386)
- **PLS** demonstrated inherent robustness to raw spectral noise
- **ANN performance** was limited by dataset size (n = 125 samples)
