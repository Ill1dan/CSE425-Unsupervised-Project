# 🎵 Unsupervised Learning Project  
## VAE-Based Music Clustering (English vs Bangla)

---

## 📌 Project Overview

This project explores **unsupervised music clustering** using audio features extracted from songs in **English** and **Bangla**.  
The goal is to analyze whether music from different languages forms distinguishable clusters based purely on audio characteristics.

Before applying deep learning models, a **classical baseline clustering pipeline** is implemented using traditional machine learning techniques.  
A **Variational Autoencoder (VAE)** is then trained to learn compact latent representations of music segments, and clustering performance in the latent space is compared against the baseline.

The project focuses on:
- Audio preprocessing
- Feature extraction
- Baseline clustering (PCA + K-Means)
- Latent space learning using VAE
- Unsupervised clustering and evaluation

---

## 🎧 Dataset Description

- **Languages:** English, Bangla  
- **Samples:**  
  - 100 English songs  
  - 100 Bangla songs  
- **Segment length:** 30 seconds  
- **Sampling rate:** 22,050 Hz  
- **Audio format:** Mono WAV  

> Raw audio files are intentionally **excluded from Git** due to size and copyright considerations.

---

## 🔧 Preprocessing Pipeline

Implemented in `src/preprocess_audio.py`:

1. Load audio using `librosa`
2. Convert stereo audio to mono
3. Resample to 22,050 Hz
4. Extract a **30-second segment from the middle** of each song  
   (to avoid long intros and silence)
5. Save standardized WAV files to `data/processed/`

✔ Ensures consistent input length  
✔ Reduces bias caused by instrumental intros  

---

## 🧾 Metadata Generation

`src/generate_metadata.py` creates `data/metadata.csv`, which acts as the **single source of truth** for the dataset.

Each row contains:

| Column    | Description |
|----------|-------------|
| id       | Unique audio identifier |
| filepath | Path to processed audio |
| language | English / Bangla |
| title    | (unused) |
| lyrics   | (unused) |

---

## 📊 Baseline Clustering (Phase 3)

Before training the VAE, a **classical unsupervised baseline** is established to provide a fair comparison.

### Feature Representation
- **MFCC (flattened)** audio features
- Fixed-length vectors extracted from 30-second audio segments
- Standardized using a global scaler

### Baseline Pipeline
1. Load flattened MFCC feature vectors
2. Apply **PCA** for dimensionality reduction (64 dimensions)
3. Perform **K-Means clustering** with `k = 2`
4. Evaluate clustering quality using standard metrics
5. Visualize cluster structure in 2D

### Evaluation Metrics
- **Silhouette Score**
- **Calinski–Harabasz Index**

### Visualization
- **t-SNE** scatter plot of PCA-reduced features
- Each point represents one song
- Colors indicate cluster assignments

### Outputs
- `results/clustering_metrics.csv`
- `results/baseline_cluster_assignments.csv`
- `results/latent_visualization/baseline_tsne_pca64_k2.png`

This baseline result is later used as a reference to evaluate the effectiveness of the VAE-based clustering.

---

## 🧠 Model Architecture

### Variational Autoencoder (VAE)

- **Input:** Audio features (MFCC or Mel-spectrogram)
- **Encoder:** Fully connected layers → latent mean & variance
- **Latent space:** Low-dimensional continuous space
- **Decoder:** Reconstructs the input features
- **Loss function:** Reconstruction loss + KL divergence

The learned latent vectors are extracted and used for clustering and visualization.

---

## 📈 Clustering & Evaluation

Clustering is performed on:
- Baseline PCA-reduced MFCC features
- VAE latent representations

### Algorithms
- K-Means clustering

### Metrics
- Silhouette Score
- Calinski–Harabasz Index

### Visualizations
- t-SNE plots for baseline features
- t-SNE plots for VAE latent space

All results are saved under: