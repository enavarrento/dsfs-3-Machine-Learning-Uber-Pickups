# Uber Pickups - Spatial Clustering Project

## Project Overview
This repository contains a machine learning pipeline designed to identify high-density pickup zones (hot-zones) for Uber drivers in New York City. By clustering geographical coordinate data over specific temporal windows, the project provides actionable insights to optimize fleet dispatching.

This project validates Bloc 3: Machine Learning (Structured Data) for the RNCP35288 Certification at Jedha School.

## Methodology & Algorithms
An Agile/MVP methodology was utilized. Initial prototyping isolated data to a single day and hour to establish a baseline before generalizing the analysis.

1. **KMeans (Centroid-based):** Used to partition the city into fixed, roughly spherical dispatch hubs. Highly computationally efficient.
2. **DBSCAN (Density-based):** Used to identify naturally occurring, high-density hot-zones. This model successfully mapped organic urban density but highlighted the "chaining effect" inherent to tightly packed environments like Manhattan.

**Evaluation Bias Note:** While KMeans achieved a higher Silhouette Score (0.392 vs. 0.209), this metric inherently favors spherical clusters. DBSCAN's irregular, density-driven shapes are mathematically penalized by the Silhouette metric, despite often providing a more realistic map of actual street-level demand.

## Data Scope (2014 vs. 2015)
This MVP relies strictly on the **2014 dataset**, which contains exact continuous variables (`Lat`, `Lon`) required for unsupervised spatial clustering. 

The 2015 dataset was intentionally excluded from this iteration. The 2015 data replaces exact coordinates with categorical `LocationID`s mapped to predefined TLC (Taxi and Limousine Commission) zones. Applying unsupervised clustering algorithms to data that has already been pre-clustered into administrative polygons is mathematically redundant. Future iterations of this project will involve geospatial joins (centroid mapping) or transitioning to supervised time-series forecasting to utilize the 2015 data.

## Repository Structure
```text
├── assets/                 # Exported HTML maps and visualizations
├── data/
│   ├── raw/                # Original Uber trip datasets (not tracked in Git)
│   └── processed/          # Cleaned datasets
├── notebooks/              # Jupyter notebooks for prototyping and presentation
│   ├── 01-Uber_Pickups.ipynb              # Project requirements
│   ├── 02-uber-pickups-prototype.ipynb    # Initial exploration
│   └── 03-uber-pickups-presentation.ipynb # Final pipeline and evaluation
├── src/                    # Reusable Python modules
│   ├── preprocessing.py    # Data ingestion, consolidation, and temporal filtering
│   └── clustering.py       # KMeans and DBSCAN models
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Environment Setup
This project leverages Intel-optimized architectures for accelerated machine learning execution over millions of rows.

1. Install dependencies:
   \`pip install -r requirements.txt\`
2. Ensure Intel optimizations are triggered at the start of any execution script:
   ```python
   from sklearnex import patch_sklearn
   patch_sklearn()
   ```