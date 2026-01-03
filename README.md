# 🎵 Unsupervised Learning Project  
## VAE-Based Music Clustering (English vs Bangla)

---

## 📌 Project Overview

This project explores **unsupervised music clustering** using audio features extracted from songs in **English** and **Bangla**.  
A **Variational Autoencoder (VAE)** is trained to learn compact latent representations of music segments, which are later clustered to analyze cross-language musical similarities.

The project focuses on:
- Audio preprocessing
- Feature extraction
- Latent space learning (VAE)
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
2. Convert to mono
3. Resample to 22,050 Hz
4. Extract a **30-second segment from the middle** of each song  
   (to avoid intros and silence)
5. Save standardized WAV files to `data/processed/`

✔ Ensures consistent input length  
✔ Reduces bias from long instrumental intros  

---

## 🧾 Metadata Generation

`src/generate_metadata.py` creates: `data/metadata.csv`


Each row contains:

| Column    | Description |
|----------|-------------|
| id       | Unique index |
| filepath | Path to processed audio |
| language | English / Bangla |
| title    | (unused) |
| lyrics   | (unused) |

This file acts as the **single source of truth** for dataset loading.

---

## 🧠 Model Architecture

### Variational Autoencoder (VAE)

- **Input:** Audio features (MFCC / Mel-spectrogram)
- **Encoder:** Dense layers → latent mean & variance
- **Latent space:** Low-dimensional continuous space
- **Decoder:** Reconstructs input features
- **Loss:** Reconstruction + KL divergence

The learned latent vectors are later used for clustering.

---

## 📊 Clustering & Evaluation

- Algorithms:
  - K-Means
  - Hierarchical clustering (optional)
- Metrics:
  - Silhouette score
  - Davies–Bouldin index
- Visualizations:
  - Latent space plots
  - Cluster separation analysis

Results are saved in:

`results/clustering_metrics.csv`
`results/latent_visualization/`

---

## ▶️ How to Run the Project

### 1️⃣ Setup environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt