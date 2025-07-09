# Gauging-$\beta$: Border-aware hierarchical clustering based on density and proximity

This project implements a novel density-based clustering algorithm that automatically detects clusters with varying shapes, densities, and proximities without requiring predefined parameters like the number of clusters.

## Overview

The algorithm works by analyzing local density characteristics and utilizing a concept of "gauging proximity" between points and clusters. It can identify clusters in complex datasets where traditional algorithms might struggle.

## Citation

Jinli Yao and Yong Zeng, "Gauging-β: Border-aware density clustering with hierarchical single-linkage." *Pattern Recognition*, 2025. (accepted)

## Key Features

- Automatic detection of clusters without specifying the number of clusters beforehand
- Handling of varying cluster shapes and densities
- Border point identification and assignment
- Adaptive thresholding for determining cluster proximity
- Support for different distance metrics (euclidean and precomputed)
- Support for different linkage methods (single and average)

## Main Components

### 1. Perception Class (`gauging_proximity.py`)

The main clustering algorithm that:

- Initializes each point as its own cluster
- Merges clusters based on distance and proximity measures
- Uses adaptive thresholds to determine when to merge clusters
- Handles the assignment of border points

### 2. Density Calculation (`local_density_calculation.py`)

Identifies points with lower density that might represent borders between clusters:

- Calculates point density using a radius-based approach
- Uses quartile-based thresholding to identify lower-density points
- Provides visualization for density-based point classification

### 3. Main Application (`main.py`)

Controls the execution of the clustering algorithm on various datasets:

- Loads test data
- Applies the clustering algorithm
- Saves results and generates visualization

## Usage

```python
from gauging_proximity import Perception
import numpy as np

# Load your data
X = load_data('your_dataset.txt')

# Initialize the model
model = Perception()

# Fit the model and get cluster assignments
labels, points = model.fit(X)

# Save or visualize results
```

## Parameters

- `metric`: Distance metric ('euclidean' or 'precomputed')
- `linkage`: Linkage method for clustering ('single' or 'average')
- `log_path`: Directory for saving plots and logs

## Datasets

The algorithm has been tested on various standard clustering datasets, including:

- Simple 2D/3D geometric shapes
- Spiral datasets
- Blobs with varying densities
- Complex shapes like rings and diamonds
- Real-world datasets

## Visualization

The algorithm provides visualization capabilities to:

- Show cluster assignments
- Track the evolution of clusters
- Visualize border points and their assignments
