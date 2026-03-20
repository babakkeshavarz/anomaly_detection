# Anomaly Detection

This repository focuses on anomaly detection workflows and experimentation.

As a first step, the emphasis is on understanding and exploring the data.  
Before building anomaly detection models, it is important to establish a baseline for how different clustering algorithms behave across various data structures.

The initial objectives of this project are:

• Explore synthetic and real datasets  
• Compare clustering algorithm performance  
• Build intuition around algorithm strengths and failure modes  
• Establish baseline evaluation metrics

---

## Clustering Benchmarking

Clustering is used here primarily as an exploratory and diagnostic tool.  
Different algorithms exhibit very different behavior depending on geometry, density variation, noise, and overlap.

The benchmark currently evaluates:

• K-Means  
• Gaussian Mixture Models (GMM)  
• DBSCAN  
• HDBSCAN  
• Agglomerative Clustering  
• Spectral Clustering  
• Mean Shift

---

## Computational Complexity Overview

Understanding computational complexity is critical when selecting algorithms for realistic datasets.

**K-Means**  
O(n · k · i · d)  
Efficient and scalable, but assumes convex, spherical clusters.

**Gaussian Mixture Models (GMM)**  
O(n · k · i · d²)  
Flexible cluster shapes, but higher cost due to covariance estimation.

**DBSCAN**  
O(n log n) to O(n²)  
Density-based, sensitive to eps, struggles with varying density.

**HDBSCAN**  
O(n log n) to O(n²)  
Hierarchical density-based approach, more robust to density variation.

**Agglomerative Clustering**  
O(n²) memory O(n²)  
Captures complex shapes but limited scalability.

**Spectral Clustering**  
O(n²) memory + O(n³) decomposition  
Very powerful for complex geometry, but computationally expensive.

**Mean Shift**  
O(n² · i)  
Adaptive cluster count, often slow for larger datasets.

---

## Key Observations from Benchmarking

Experiments across multiple synthetic datasets (blobs, ellipsoids, spirals, rings, varying density, and overlap scenarios) reveal several consistent patterns:

• Partitioning methods (K-Means, GMM) perform well on convex structures  
• Graph-based methods (Spectral) can work well but require careful tuning  
• Density-based methods are generally better for irregular geometry  
• Noise handling behavior strongly influences evaluation metrics

Across most tested scenarios:

**HDBSCAN, with appropriate parameter tuning, provides the most stable and interpretable results.**

Notable advantages:

• Naturally handles varying density  
• No requirement to predefine cluster count  
• Strong noise awareness  
• Performs well on non-convex geometry  
• Consistently strong ARI / NMI performance

Important nuance:

HDBSCAN is not universally superior. Performance depends on:

• Parameter choices (min_cluster_size, min_samples)  
• Data scaling  
• Underlying density separability

However, relative to other methods, it tends to degrade more gracefully and requires fewer dataset-specific assumptions.

---

## Example Benchmark Output

<p align="center">
  <img src="images/clustering_benchmarking.png" width="650">
</p>

---


## Projection Methods Benchmarking

PCA remains the strongest overall baseline. It consistently performs well across datasets and is by far the fastest method. It preserves global variance effectively, but as a linear method, it struggles to capture non-linear structures such as spirals, concentric patterns, or interleaving manifolds.

UMAP, while sometimes struggling to preserve the exact geometry of structures such as concentric clusters, performs very well at separating clusters with different shapes and densities. It provides a strong balance between local and global structure preservation and scales efficiently to larger datasets.

t-SNE is particularly effective at visual cluster separation and captures local structure very well. However, it distorts global distances, meaning that spacing between clusters is not always meaningful. It is also computationally more expensive compared to PCA and UMAP.

Isomap performs well when the data lies on smooth non-linear manifolds, such as spiral-like structures, and is better than t-SNE at preserving global geometry. However, it is sensitive to noise and neighborhood size, and its cubic complexity makes it less practical for larger datasets.

MDS attempts to preserve pairwise distances between all points, which makes it conceptually appealing. In practice, it is computationally expensive and rarely provides better results than the other methods.

Across different dataset types, several patterns emerge. PCA performs well on simple Gaussian or blob-like structures, but fails on non-linear geometries. UMAP and t-SNE significantly improve cluster separation in anisotropic and complex datasets. For spiral and manifold structures, PCA fails entirely, while Isomap and UMAP capture the underlying structure more effectively. In concentric datasets, UMAP and t-SNE separate clusters clearly, although both may distort the true geometry. In varying-density datasets, UMAP tends to be the most stable and consistent method.

### Computational Complexity

```python
PROJECTION_COMPLEXITY_SHORT = {
    "PCA": "O(n·d^2)",
    "t-SNE": "O(i·n^2)",
    "Isomap": "O(n^3)",
    "MDS": "O(n^2–n^3)",
    "UMAP": "O(n log n)"
}

## Project Goals

This repository will progressively expand toward:

• Robust anomaly detection pipelines  
• Feature engineering strategies  
• Density and distance-based detection methods  
• Scalable implementations  
• Model evaluation and diagnostics

---

## Notes

Clustering is treated here as an exploratory lens rather than a final modeling solution.

Understanding cluster structure, separability, density variation, and algorithm behavior provides critical intuition for downstream anomaly detection tasks.