"""
Compute cluster prototypes (medoids) for visualization and exploration.

A prototype is the most "representative" sample from each cluster,
typically the sample closest to the cluster centroid.
"""

try:
    import numpy as np
except ImportError:
    np = None
