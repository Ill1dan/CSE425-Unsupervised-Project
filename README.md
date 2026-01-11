# 🎵 VAE-Based Hybrid Language Music Clustering (English vs Bangla)

This repository investigates **unsupervised clustering of bilingual music** using **Variational Autoencoders (VAEs)**.  
We compare (1) a **fully-connected VAE baseline** trained on **flattened MFCC vectors**, (2) a **convolutional VAE** trained on **mel-spectrograms**, and (3) a **β-VAE regularization sweep** that exposes a common failure mode (**posterior collapse**) under strong KL pressure.

---

## ✅ Key Results

**Dataset:** 200 tracks (100 English, 100 Bangla)

### 1) Baseline: MLP-VAE on MFCC vectors
- Latent dim: **16**
- Clustering (KMeans, k=2) on latent means μ:
  - **Silhouette:** 0.19124  
  - **CH:** 21.72773  
- Interpretation: modest structure; weak separation in audio-only MFCC space.

### 2) Proposed: CNN-VAE on mel-spectrograms
- Input shape: **(200, 1, 128, 256)** (globally standardized: mean≈0, std≈1)
- Latent dim: **32**
- Clustering (KMeans, k=2) on latent means μ:
  - **Silhouette:** 0.46427  
  - **CH:** 193.148  
  - **DBI:** 0.90119 (lower is better)  
  - **ARI:** 0.12578 (labels used only for evaluation)
- Interpretation: strong improvement; mel + conv encoder learns more clusterable geometry.

**Comparison (not main): Agglomerative, k=2**
- Silhouette: 0.44378, CH: 172.491, DBI: 0.96663, ARI: 0.18118  
- Better ARI, but weaker overall geometric separation than KMeans.

### 3) Regularization analysis: β-VAE (MFCC-flat), β ∈ {1,2,4,6}
- Latent dim: **32**
- Observation: clustering frequently collapses to **one distinct cluster** (KMeans warning).
- Internal metrics: often **undefined / not meaningful**
- Evaluation-only metrics near chance:
  - ARI ≈ 0.0, NMI ≈ 0.0096, Purity ≈ 0.505
- Interpretation: consistent with **posterior collapse** under strong KL pressure.

---

## 📌 Project Overview

Unsupervised clustering of music is challenging because raw audio features are high-dimensional and non-linear.  
This project follows a representation learning pipeline:

1. **Audio preprocessing** (standardized segments)
2. **Feature extraction**  
   - MFCC vectors (baseline + β-VAE analysis)  
   - Mel-spectrogram tensors (CNN-VAE)
3. **Representation learning** with VAEs (latent mean vectors μ)
4. **Clustering** in latent space (KMeans, plus comparisons in the CNN-VAE setting)
5. **Evaluation** via internal metrics (Silhouette, CH, DBI) and evaluation-only label metrics (ARI/NMI/Purity)

---

## 🎧 Dataset Description

- **Languages:** English, Bangla  
- **Samples:** 200 total  
  - 100 English songs  
  - 100 Bangla songs  
- **Segment length:** 30 seconds  
- **Sampling rate:** 22,050 Hz  
- **Audio format:** Mono WAV  

> Raw audio is intentionally excluded from Git (size + copyright).

---

## 🔧 Preprocessing Pipeline

Implemented in `src/preprocess_audio.py`:

1. Load audio with `librosa`
2. Convert stereo → mono
3. Resample to 22,050 Hz
4. Extract a **30-second segment from the middle** of each track  
   (reduces bias from long intros/silence)
5. Save standardized WAV files to `data/processed/`

✅ Ensures consistent input length  
✅ Stabilizes feature dimensionality  

---

## 🧾 Metadata Generation

`src/generate_metadata.py` produces `data/metadata.csv` as the dataset index.

| Column    | Description |
|----------|-------------|
| id       | Unique audio identifier |
| filepath | Path to processed audio |
| language | English / Bangla |
| title    | (optional / unused) |
| lyrics   | (optional / unused) |

---

## 🧪 Experimental Setup (Feature Representations)

We study two audio feature representations.

### 1) MFCC vectors (baseline)
- Flattened MFCC features per track  
- Example shape check: **(200, 10240)** = 40 MFCC × 256 frames
- Global standardization applied:
  - mean ≈ 0, std ≈ 1
  - stable min/max range (no extreme outliers)

**Why MFCC is a baseline:**  
Raw MFCC space shows substantial overlap between English and Bangla tracks, indicating weak linear separability and motivating non-linear representation learning.

### 2) Mel-spectrogram tensors (proposed)
- Log-mel spectrograms preserving time–frequency structure  
- Input tensor: **(200, 1, 128, 256)**  
- Globally standardized to zero mean and unit variance

---

## 🧠 Models

### A) Baseline MLP-VAE (MFCC)
- Fully connected encoder/decoder
- Trained for **50 epochs**
- Small KL weight (**β = 0.0001**) to prioritize reconstruction and avoid early collapse
- Latent mean vectors extracted:
  - **μ ∈ R¹⁶**

### B) CNN-VAE (mel-spectrogram)
- Convolutional encoder/decoder over 2D time–frequency maps
- KL warmup (annealing): β ramped **0.1 → 1.0**
- GPU training supported (CUDA)
- Latent mean vectors extracted:
  - **μ ∈ R³²**

### C) β-VAE (MFCC) — regularization analysis
- Same MFCC-flat input representation
- Trained with fixed β ∈ {1,2,4,6}
- Used to study stability and collapse in latent space

---

## 📈 Clustering & Evaluation

Clustering is performed on **latent mean vectors μ**.

### Algorithms
- **KMeans** (primary; k=2 for two-language grouping)
- **Agglomerative** (comparison in CNN-VAE setting)
- **DBSCAN** (exploratory; unstable here)

### Metrics (unsupervised structure)
- **Silhouette Score** (higher is better)
- **Calinski–Harabasz (CH)** (higher is better)
- **Davies–Bouldin (DBI)** (lower is better)

### Evaluation-only (uses language labels, not training)
- **ARI**, **NMI**, **Purity**

---

## 📁 Outputs / Artifacts

All outputs are stored under `results/`:

### Easy / Baseline MLP-VAE (MFCC)
- `results/easy/models/vae_mlp.pt`
- `results/easy/logs/vae_train_log.csv`
- `results/easy/latents/latent_mu.npy`
- `results/easy/clustering/clustering_metrics_k2.csv`
- `results/easy/latent_visualization/vae_loss_curves.png`
- `results/easy/latent_visualization/pca_k2.png`
- `results/easy/latent_visualization/tsne_k2.png`

### Medium / CNN-VAE (mel)
- `results/medium/models/cnn_vae.pt`
- `results/medium/latents/` (μ vectors)
- `results/medium/metrics/clustering_metrics.csv`
- `results/medium/visualizations/`
  - `pca_kmeans_k2.png`
  - `tsne_kmeans_k2.png`
  - `pca_language.png`
  - `tsne_language.png`

### Hard / β-VAE analysis
- `results/hard/latents/` (per β)
- `results/hard/metrics/clustering_metrics_beta.csv`
- `results/hard/visualizations/`
  - `latent_pca_beta_1.0.png`, `latent_tsne_beta_1.0.png`
  - `latent_pca_beta_2.0.png`, `latent_tsne_beta_2.0.png`
  - `latent_pca_beta_4.0.png`, `latent_tsne_beta_4.0.png`
  - `latent_pca_beta_6.0.png`, `latent_tsne_beta_6.0.png`
- `results/hard/reconstructions/` (reconstruction comparison plots)

---