import numpy as np
import matplotlib.pyplot as plt 
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial.distance import pdist, squareform

# Generate some random points in d-dimensional space (d=1000)

np.random.seed(31)
n_points = 500
d = 1000
X = np.random.randn(n_points, d) # n points with d dimensions

# Calculating pairwise distances
distances = pdist(X, metric='euclidean')

# Now for projecting to lower k-dimension (k=100)
k = 100
proj = GaussianRandomProjection(n_components=k)
X_proj = proj.fit_transform(X)

# Calculate pairwise lower dimension distances
proj_distances = pdist(X_proj, metric='euclidean')



plt.figure(figsize=(15, 15))
plt.scatter(distances, proj_distances, alpha=0.3, c=distances, cmap="viridis", edgecolors='k', s=10)
plt.plot([min(distances), max(distances)], [min(distances), max(distances)], 'r--', lw=2, label="Ideal Preservation (y=x)") # Add diagonal reference line (y = x)

# Scale axes consistently
max_val = max(distances.max(), proj_distances.max())
plt.xlim(0,max_val)
plt.ylim(0,max_val)

plt.xlabel("Original Distances (High Dim)")
plt.ylabel("Projected Distances (Low Dim)")
plt.title("Johnson-Lindenstrauss Lemma: Distance Preservation")
plt.legend()
plt.colorbar(label="Original Distance Magnitude")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
