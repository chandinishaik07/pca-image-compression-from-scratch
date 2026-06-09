# PCA + Image Compression from Scratch

Implemented PCA and image compression using only NumPy — zero sklearn for core math.  

---

## What this project does

Takes high-dimensional image data and compresses it using **Singular Value Decomposition (SVD)**, 
keeping only the most important directions of variance. Every line of math is implemented from scratch.

**Key result:** A 280×280 image compressed **7x smaller** at near-original visual quality (PSNR = 38 dB) using k=20 singular values.

---

## Math implemented from scratch

| Concept | Formula | File |
|---|---|---|
| Mean centering | `X = X - μ` | `pca_compression.py` |
| Covariance matrix | `C = XᵀX / (n-1)` | `pca_compression.py` |
| SVD decomposition | `A = UΣVᵀ` | `pca_compression.py` |
| Variance explained | `λᵢ = σᵢ² / (n-1)` | `pca_compression.py` |
| PCA projection | `Z = X @ Vᵀ[:k].T` | `pca_compression.py` |
| Reconstruction | `X̂ = Z @ Vᵀ[:k] + μ` | `pca_compression.py` |
| Truncated SVD | `A_k = U[:,:k] Σ[:k] Vᵀ[:k,:]` | `pca_compression.py` |

> sklearn is used **only** to load MNIST. All math is pure NumPy.

---

## Results

### Variance Explained
- **149 components** explain 95% of variance in MNIST (out of 784 dimensions)
- Top 10 components alone capture **48.9%** of total variance

### Image Compression (280×280 image)

| k | PSNR (dB) | Compression Ratio | Quality |
|---|---|---|---|
| 5 | 23.0 | 28x | Blurry but recognizable |
| 20 | 38.0 | 7x | Near-original |
| 50 | 40.2 | 2.8x | Excellent |
| 100 | 44.1 | 1.4x | Near-perfect |

### Math Verification
- SVD reconstruction error: `5.40e-13` (machine precision)
- Principal component orthogonality: `2.00e-15` (perfectly orthogonal)
- Eigenvalues from SVD match direct decomposition: `True`

---

## Visualizations

![PCA Results]<img src="images/pca_results.png" width="1000">

**Top row:** Variance explained curve · Singular value decay · 2D MNIST projection  
**Middle row:** Top 8 eigendigits (principal components) · Image reconstruction at k=5,20,50,100  
**Bottom row:** PSNR vs k · Compression ratio vs k · Per-component variance

---

## How to run

```bash
pip install numpy matplotlib scikit-learn pandas
python pca_compression.py
```

Output: `pca_results.png` with all visualizations + math verification printed to console.

---

## Key concepts explained

**Why covariance matrix?**  
`C = XᵀX / (n-1)` captures how every pair of features (pixels) varies together. 
Diagonal = individual variances. Off-diagonal = co-movement between pixels.

**Why SVD instead of eigendecomposition?**  
Both give the same result, but SVD on X is numerically more stable than 
eigendecomposition on XᵀX. Industry standard for large-scale PCA.

**What does truncating singular values mean?**  
The Eckart-Young theorem proves that keeping only the top k singular values 
gives the best possible rank-k approximation — you cannot do better with k numbers.

**Why does compression work?**  
Natural images have highly correlated pixels (neighboring pixels tend to be similar). 
SVD finds and exploits this redundancy. The singular values decay fast — 
the tail contributes almost nothing, so we discard it.

---

## Dataset
- **MNIST** — 70,000 handwritten digit images, 28×28 pixels each
- Used 5,000 samples for demonstration (full 70k supported)

## Tech stack
- Python 3.11
- NumPy (all core math)
- Matplotlib (all visualizations)
- scikit-learn (data loading only)

---
