# Transformer-Based Optimal Sensor Placement for Structural Health Monitoring of Probe Cards

This repository contains code and materials for our **Transformer-based** approach to **Optimal Sensor Placement (OSP)** and **mechanical failure detection** (baseline / loose screw / crack) on semiconductor **probe cards** using Frequency Response Function (FRF) data. The method combines a CNN encoder with a Transformer and attention to both classify failures and highlight informative sensor locations.

> **Publication status**: The paper has been **submitted and is under review**.  
> A **preprint** is available on arXiv: https://arxiv.org/abs/2509.07603  
> Please **cite the preprint** for now. The official journal link will be added after publication.

---

## Highlights

- **Hybrid CNN + Transformer** with attention for interpretable OSP. :contentReference[oaicite:3]{index=3}  
- Dataset comprises **28 sensor channels × 150 frequency points** per sample; total **3,750 samples** after physics-informed expansion across material/temperature/loading cases. :contentReference[oaicite:4]{index=4}  
- The training script performs **SMOTE** + physics-aware augmentations, **grid search**, and a **rigorous CV** pipeline. :contentReference[oaicite:5]{index=5}  
- Example run achieved **~0.954 balanced accuracy** in grid search (best config). :contentReference[oaicite:6]{index=6}

---

## Repository structure

