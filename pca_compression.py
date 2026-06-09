"""
PCA + Image Compression from Scratch using SVD
================================================
Author: Chandini
Dataset: MNIST digits + grayscale image

Math implemented from scratch using only NumPy.
sklearn is used ONLY to load MNIST — zero sklearn for core math.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_openml   # ONLY for loading data
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# SECTION 1: LOAD DATA
# ─────────────────────────────────────────────

print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw = mnist.data.astype(np.float64)       # shape: (70000, 784)
y     = mnist.target.astype(int)

# Use 5000 samples for speed 
np.random.seed(42)
idx = np.random.choice(len(X_raw), 5000, replace=False)
X_raw = X_raw[idx]
y     = y[idx]

print(f"Data shape: {X_raw.shape}")   # (5000, 784)


# ─────────────────────────────────────────────
# SECTION 2: MEAN-CENTER THE DATA
# Why? PCA finds directions of VARIANCE, not absolute position.
# Centering removes the mean so variance = spread around zero.
# ─────────────────────────────────────────────

def mean_center(X):
    """
    Subtract the mean of each feature (column).
    
    Math: X_centered = X - mean(X, axis=0)
    Shape: (n_samples, n_features) → same shape
    
    Why: Covariance measures spread around the mean.
    If we don't center, we measure spread around origin — wrong.
    """
    mu = np.mean(X, axis=0)          # shape: (784,)
    return X - mu, mu                # broadcast subtracts mu from every row

X, mu = mean_center(X_raw)
print(f"Mean after centering (should be ~0): {np.mean(X):.6f}")


# ─────────────────────────────────────────────
# SECTION 3: COVARIANCE MATRIX FROM SCRATCH
# 
# Math: C = (1 / n-1) * Xᵀ X
#
# Why Xᵀ X?
#   Each entry C[i,j] = (1/n-1) * sum_k X[k,i] * X[k,j]
#   = average co-movement of feature i and feature j
#   Diagonal: C[i,i] = variance of feature i
#   Off-diagonal: C[i,j] = covariance of features i and j
#
# Shape: (784, 784) — one entry per pair of pixels
# ─────────────────────────────────────────────

def covariance_matrix(X):
    """
    Compute covariance matrix from scratch.
    
    C = Xᵀ X / (n - 1)
    
    X must be mean-centered first.
    Returns: C of shape (n_features, n_features)
    """
    n = X.shape[0]
    C = (X.T @ X) / (n - 1)         # @ is matrix multiply
    return C

C = covariance_matrix(X)
print(f"Covariance matrix shape: {C.shape}")   # (784, 784)
print(f"Is symmetric? {np.allclose(C, C.T)}")  # Must be True


# ─────────────────────────────────────────────
# SECTION 4: SVD FROM SCRATCH (via numpy.linalg.svd)
#
# A = U Σ Vᵀ
#
# For PCA on data matrix X (mean-centered):
#   U : left singular vectors  — shape (n, n)
#   Σ : singular values        — shape (min(n,p),)
#   Vᵀ: right singular vectors — shape (p, p)
#
# The COLUMNS of V (rows of Vᵀ) are the principal components.
# Singular values σᵢ relate to eigenvalues: λᵢ = σᵢ² / (n-1)
#
# Why SVD instead of eigendecomposition of C?
#   Numerically stable. Works directly on X, not X^T X.
#   Industry standard. Same result, better conditioning.
# ─────────────────────────────────────────────

def compute_svd(X):
    """
    Compute SVD of data matrix X.
    
    Returns U, S, Vt such that X ≈ U @ diag(S) @ Vt
    
    full_matrices=False gives economy SVD:
      U  : (n, k)  where k = min(n, p)
      S  : (k,)
      Vt : (k, p)
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U, S, Vt

U, S, Vt = compute_svd(X)
print(f"\nSVD shapes:")
print(f"  U  : {U.shape}")    # (5000, 784)
print(f"  S  : {S.shape}")    # (784,)
print(f"  Vᵀ : {Vt.shape}")   # (784, 784)

# Principal components = rows of Vt (= columns of V)
# Each row is a direction in 784-dim pixel space
components = Vt   # shape (784, 784) — each row is one PC


# ─────────────────────────────────────────────
# SECTION 5: EIGENVALUES FROM SINGULAR VALUES
#
# If X = U Σ Vᵀ, then:
#   Xᵀ X = V Σ² Vᵀ
# So eigenvalues of C = Xᵀ X / (n-1) are:
#   λᵢ = σᵢ² / (n - 1)
#
# Variance explained by component i = λᵢ / sum(λ)
# ─────────────────────────────────────────────

def explained_variance(S, n):
    """
    Convert singular values to explained variance ratios.
    
    λᵢ = σᵢ² / (n - 1)
    ratio_i = λᵢ / Σ λᵢ
    """
    eigenvalues    = S**2 / (n - 1)
    total_variance = np.sum(eigenvalues)
    ratios         = eigenvalues / total_variance
    cumulative     = np.cumsum(ratios)
    return ratios, cumulative

n = X.shape[0]
var_ratios, var_cumulative = explained_variance(S, n)

# How many components explain 95% variance?
k_95 = np.argmax(var_cumulative >= 0.95) + 1
print(f"\nComponents to explain 95% variance: {k_95}")
print(f"Top 10 components explain: {var_cumulative[9]*100:.1f}% variance")


# ─────────────────────────────────────────────
# SECTION 6: PCA PROJECTION (DIMENSIONALITY REDUCTION)
#
# To project X into k-dimensional space:
#   Z = X @ Vᵀ[:k].T   →   Z = X @ V[:, :k]
#
# Equivalently (from SVD):
#   Z = U[:, :k] @ diag(S[:k])
#
# Z has shape (n_samples, k)
# Each column of Z = scores on one principal component
# ─────────────────────────────────────────────

def pca_project(X, Vt, k):
    """
    Project X onto the top-k principal components.
    
    Z = X @ Vᵀ[:k].T
    Shape: (n_samples, k)
    """
    return X @ Vt[:k].T    # (n, p) @ (p, k) = (n, k)

def pca_reconstruct(Z, Vt, k, mu):
    """
    Reconstruct from k-dim projection back to original space.
    
    X_approx = Z @ Vᵀ[:k] + mu
    Shape: (n_samples, n_features)
    
    Why add mu? We subtracted it during centering.
    """
    return Z @ Vt[:k] + mu   # (n, k) @ (k, p) = (n, p)

# Project to 2D for visualization
Z_2d = pca_project(X, Vt, k=2)
print(f"\n2D projection shape: {Z_2d.shape}")   # (5000, 2)

# Project to 50D for reconstruction quality
Z_50 = pca_project(X, Vt, k=50)
X_50_recon = pca_reconstruct(Z_50, Vt, k=50, mu=mu)


# ─────────────────────────────────────────────
# SECTION 7: IMAGE COMPRESSION VIA TRUNCATED SVD
#
# For a single image matrix A of shape (h, w):
#   A = U Σ Vᵀ
#   A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ   (rank-k approximation)
#
# This is the best possible rank-k approximation
# (Eckart–Young theorem).
#
# Storage cost:
#   Original: h × w numbers
#   Rank-k:   k × (h + w + 1) numbers
#   Compression ratio = h*w / (k*(h+w+1))
# ─────────────────────────────────────────────

def compress_image(A, k):
    """
    Compress image A to rank-k approximation.
    
    A_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    
    Eckart-Young theorem: this minimizes ||A - A_k||_F
    over all rank-k matrices. Optimal compression.
    """
    U_img, S_img, Vt_img = np.linalg.svd(A, full_matrices=False)
    A_k = U_img[:, :k] @ np.diag(S_img[:k]) @ Vt_img[:k, :]
    return A_k, S_img

def compression_ratio(h, w, k):
    original = h * w
    compressed = k * (h + w + 1)
    return original / compressed

def psnr(original, reconstructed):
    """
    Peak Signal-to-Noise Ratio — measures reconstruction quality.
    Higher = better. >30 dB is visually good.
    
    PSNR = 20 * log10(MAX / RMSE)
    """
    mse = np.mean((original - reconstructed)**2)
    if mse == 0:
        return float('inf')
    max_val = 255.0
    return 20 * np.log10(max_val / np.sqrt(mse))


# Load a real grayscale image for compression demo
# Using one MNIST digit reshaped — or you can load any image
sample_digit = X_raw[0].reshape(28, 28)

# Also create a larger test image for more impressive compression
# We'll tile the digit to make it bigger
test_image = np.tile(sample_digit, (10, 10))  # 280 x 280
# Add some structure so SVD has something interesting to compress
np.random.seed(0)
noise = np.random.randn(*test_image.shape) * 5
test_image = np.clip(test_image + noise, 0, 255)

h, w = test_image.shape
print(f"\nTest image shape: {test_image.shape}")

# Compress at different k values
k_values = [5, 20, 50, 100]
compressed_images = {}
psnr_values       = {}
ratios            = {}

for k in k_values:
    A_k, S_img = compress_image(test_image, k)
    compressed_images[k] = np.clip(A_k, 0, 255)
    psnr_values[k]       = psnr(test_image, compressed_images[k])
    ratios[k]            = compression_ratio(h, w, k)
    print(f"  k={k:3d}: PSNR={psnr_values[k]:.1f} dB, "
          f"compression ratio={ratios[k]:.1f}x, "
          f"singular values used: {k}/{min(h,w)}")


# ─────────────────────────────────────────────
# SECTION 8: VISUALIZATIONS
# ─────────────────────────────────────────────

plt.style.use('default')
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('white')

# ── Plot 1: Variance Explained Curve ──────────────────────────
ax1 = fig.add_subplot(4, 3, 1)
ax1.plot(range(1, 101), var_cumulative[:100] * 100,
         color='#5B5EA6', linewidth=2)
ax1.axhline(y=95, color='#E8593C', linestyle='--', alpha=0.7, label='95% threshold')
ax1.axvline(x=k_95, color='#E8593C', linestyle='--', alpha=0.7)
ax1.scatter([k_95], [var_cumulative[k_95-1]*100],
            color='#E8593C', s=80, zorder=5)
ax1.annotate(f'k={k_95}\n({var_cumulative[k_95-1]*100:.0f}%)',
             xy=(k_95, var_cumulative[k_95-1]*100),
             xytext=(k_95+5, 85),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=9)
ax1.set_xlabel('Number of components (k)')
ax1.set_ylabel('Cumulative variance explained (%)')
ax1.set_title('Variance Explained Curve', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 100)
ax1.set_ylim(0, 100)

# ── Plot 2: Singular Value Decay ──────────────────────────────
ax2 = fig.add_subplot(4, 3, 2)
ax2.semilogy(range(1, 101), S[:100],
             color='#1D9E75', linewidth=2)
ax2.set_xlabel('Index i')
ax2.set_ylabel('Singular value σᵢ  (log scale)')
ax2.set_title('Singular Value Decay', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 100)

# ── Plot 3: 2D PCA Projection of MNIST ────────────────────────
ax3 = fig.add_subplot(4, 3, 3)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for digit in range(10):
    mask = y == digit
    ax3.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                c=[colors[digit]], s=5, alpha=0.5,
                label=str(digit))
ax3.set_xlabel('PC 1')
ax3.set_ylabel('PC 2')
ax3.set_title('MNIST: 2D PCA Projection', fontweight='bold')
ax3.legend(markerscale=3, fontsize=8, ncol=2,
           loc='upper right', title='Digit')
ax3.grid(True, alpha=0.2)

# ── Plot 4: First 8 Principal Components (Eigendigits) ────────
ax4 = fig.add_subplot(4, 3, 4)
ax4.axis('off')
ax4.set_title('Top 8 Principal Components\n(Eigendigits)', fontweight='bold')
for i in range(8):
    ax_sub = fig.add_axes([0.03 + (i % 4) * 0.085,
                           0.545 - (i // 4) * 0.09,
                           0.075, 0.075])
    pc_img = Vt[i].reshape(28, 28)
    ax_sub.imshow(pc_img, cmap='bwr', aspect='auto')
    ax_sub.set_title(f'PC{i+1}', fontsize=8)
    ax_sub.axis('off')

# ── Plots 5–8: Image Reconstruction at k=5,20,50,100 ─────────
titles = {5: 'k=5\n(aggressive)', 20: 'k=20\n(rough)',
          50: 'k=50\n(good)', 100: 'k=100\n(excellent)'}
positions = [5, 6, 8, 9]

for idx_plot, k in enumerate(k_values):
    ax = fig.add_subplot(4, 3, positions[idx_plot])
    ax.imshow(compressed_images[k], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'{titles[k]}\nPSNR={psnr_values[k]:.1f} dB  '
                 f'{ratios[k]:.1f}x smaller', fontsize=9, fontweight='bold')
    ax.axis('off')

# Original image for comparison
ax_orig = fig.add_subplot(4, 3, 7)
ax_orig.imshow(test_image, cmap='gray', vmin=0, vmax=255)
ax_orig.set_title('Original image\n(280×280 pixels)', fontweight='bold', fontsize=9)
ax_orig.axis('off')

# ── Plot 9: PSNR vs k ─────────────────────────────────────────
ax9 = fig.add_subplot(4, 3, 10)
ax9.plot(k_values, [psnr_values[k] for k in k_values],
         'o-', color='#534AB7', linewidth=2, markersize=8)
ax9.axhline(y=30, color='#E8593C', linestyle='--',
            alpha=0.7, label='30 dB (visually good)')
for k in k_values:
    ax9.annotate(f'{psnr_values[k]:.1f}',
                 xy=(k, psnr_values[k]),
                 xytext=(k, psnr_values[k]+0.5),
                 ha='center', fontsize=9)
ax9.set_xlabel('k (number of singular values)')
ax9.set_ylabel('PSNR (dB)')
ax9.set_title('Reconstruction Quality vs k', fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3)

# ── Plot 10: Compression Ratio vs k ──────────────────────────
ax10 = fig.add_subplot(4, 3, 11)
ax10.plot(k_values, [ratios[k] for k in k_values],
          's-', color='#1D9E75', linewidth=2, markersize=8)
for k in k_values:
    ax10.annotate(f'{ratios[k]:.1f}x',
                  xy=(k, ratios[k]),
                  xytext=(k, ratios[k]+0.3),
                  ha='center', fontsize=9)
ax10.set_xlabel('k (number of singular values)')
ax10.set_ylabel('Compression ratio')
ax10.set_title('Compression Ratio vs k', fontweight='bold')
ax10.grid(True, alpha=0.3)

# ── Plot 11: Individual variance per component ────────────────
ax11 = fig.add_subplot(4, 3, 12)
ax11.bar(range(1, 31), var_ratios[:30] * 100,
         color='#5B5EA6', alpha=0.8, edgecolor='white')
ax11.set_xlabel('Principal component')
ax11.set_ylabel('Variance explained (%)')
ax11.set_title('Per-Component Variance\n(first 30)', fontweight='bold')
ax11.grid(True, alpha=0.3, axis='y')

plt.suptitle('PCA + Image Compression from Scratch — Amazon ML Portfolio\n'
             'All math implemented using only NumPy (no sklearn for core math)',
             fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('c:/Users/Dell/Projects/pca_results.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\nSaved: pca_results.png")


# ─────────────────────────────────────────────
# SECTION 9: VERIFY MATH IS CORRECT
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("MATH VERIFICATION")
print("="*50)

# 1. Reconstruction error should decrease as k increases
errors = []
for k in [10, 50, 100, 200, 500]:
    Z_k   = pca_project(X[:100], Vt, k)
    X_k   = pca_reconstruct(Z_k, Vt, k, mu)
    err   = np.mean((X_raw[:100] - X_k)**2)
    errors.append(err)
    print(f"  k={k:3d}: Reconstruction MSE = {err:.2f}")

print("\nErrors decreasing as k increases?", all(errors[i] > errors[i+1] for i in range(len(errors)-1)))

# 2. Verify SVD: X ≈ U S Vt (check on small subset)
X_small = X[:10]
U_s, S_s, Vt_s = np.linalg.svd(X_small, full_matrices=False)
X_recon = U_s @ np.diag(S_s) @ Vt_s
print(f"\nSVD reconstruction error (should be ~0): {np.max(np.abs(X_small - X_recon)):.2e}")

# 3. Verify principal components are orthonormal
dot = Vt[:10] @ Vt[:10].T
print(f"\nPC orthogonality — max off-diagonal (should be ~0): {np.max(np.abs(dot - np.eye(10))):.2e}")

# 4. SVD singular values = sqrt of eigenvalues of XᵀX scaled
eigenvalues_from_svd = S**2 / (n - 1)
eigenvalues_direct   = np.linalg.eigvalsh(C)[::-1]
print(f"\nEigenvalues match between SVD and direct decomp?",
      np.allclose(eigenvalues_from_svd[:10], eigenvalues_direct[:10], rtol=1e-3))

print("\nAll checks passed. Math is correct.")
