#!/usr/bin/env python3
"""Worker: compute single importance method. Called by extract_correlation_matrix.py."""
import sys
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def main():
    X_path, y_path, method, output_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    X = np.load(X_path)
    y = np.load(y_path)

    from kirby.hybrid import compute_feature_importance
    rep_dict = {'features': X}
    scores = compute_feature_importance(rep_dict, y, method=method)
    np.save(output_path, scores['features'])

if __name__ == '__main__':
    main()
