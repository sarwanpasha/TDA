"""margin.py after a uniform sub-pixel jitter of every centroid, to separate the margins
of the imaged tissue from those of the pixel grid the centroids were read from.

python margin_jitter.py <cohort> [amplitude in micrometers, default 0.5]
"""
import sys
import numpy as np
import margin

AMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
_rng = np.random.default_rng(0)
_orig = margin.incircle_margin


def jittered(X):
    return _orig(X + _rng.uniform(-AMP, AMP, X.shape))


margin.incircle_margin = jittered

if __name__ == '__main__':
    sys.argv = sys.argv[:2]
    margin.main()
