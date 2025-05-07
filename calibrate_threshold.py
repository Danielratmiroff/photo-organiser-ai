#!/usr/bin/env python3
"""
calibrate_threshold.py

Load face embeddings from a CSV (with columns: image_id, face_id, face_index, identity, emb_*),
compute pairwise distances and ground truth labels based on the 'identity' column,
plot the precision-recall curve vs threshold, select the optimal threshold (max F1-score),
and save this threshold into a YAML config file.

Requires:
    pip install pandas scikit-learn matplotlib pyyaml
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import yaml


def load_embeddings(csv_path, label_col):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        logging.error(f"Label column '{label_col}' not found in {csv_path}.")
        sys.exit(1)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    if not emb_cols:
        logging.error(f"No embedding columns (emb_*) found in {csv_path}.")
        sys.exit(1)
    embeddings = df[emb_cols].values
    labels = df[label_col].values
    return embeddings, labels


def compute_pairwise(embeddings, labels, sample_neg=None, random_state=None):
    n = len(embeddings)
    pos_pairs = []
    neg_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                pos_pairs.append((i, j))
            else:
                neg_pairs.append((i, j))
    if sample_neg and sample_neg < len(neg_pairs):
        rng = np.random.RandomState(random_state)
        neg_pairs = list(rng.choice(neg_pairs, size=sample_neg, replace=False))
    pairs = pos_pairs + neg_pairs
    y_true = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
    dists = np.array([np.linalg.norm(embeddings[i] - embeddings[j])
                     for i, j in pairs])
    return dists, y_true


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate face-match threshold via precision-recall curve"
    )
    parser.add_argument('embeddings_csv', type=Path,
                        help="Path to embeddings CSV file"
                        )
    parser.add_argument('--label-column', type=str, default='identity',
                        help="Column name in CSV for true identity labels")
    parser.add_argument('-o', '--output-plot', type=Path, default=Path('pr_curve.png'),
                        help="Path to save the precision-recall curve plot")
    parser.add_argument('-c', '--config', type=Path, default=Path('face_match_config.yaml'),
                        help="Path to output YAML config file")
    parser.add_argument('--sample-neg', type=int, default=None,
                        help="Number of negative pairs to sample (for speed)")
    parser.add_argument('--random-state', type=int, default=42,
                        help="Random seed for sampling negatives")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    if not args.embeddings_csv.is_file():
        logging.error(
            f"Embeddings CSV '{args.embeddings_csv}' does not exist.")
        sys.exit(1)

    logging.info(f"Loading embeddings from {args.embeddings_csv}")
    embeddings, labels = load_embeddings(
        args.embeddings_csv, args.label_column)

    logging.info("Computing pairwise distances and labels")
    dists, y_true = compute_pairwise(
        embeddings,
        labels,
        sample_neg=args.sample_neg,
        random_state=args.random_state
    )

    logging.info("Computing precision-recall curve")
    scores = -dists
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1_scores)
    best_threshold_score = thresholds[best_idx] if best_idx < len(
        thresholds) else None
    best_f1 = f1_scores[best_idx]
    distance_threshold = -best_threshold_score if best_threshold_score is not None else None

    # Compute variance of pairwise distances and log it
    dist_variance = float(np.var(dists))
    logging.info(f"Distance variance: {dist_variance:.6f}")

    logging.info(
        f"Best F1={best_f1:.3f} at distance threshold={distance_threshold:.3f}")

    plt.figure()
    plt.plot(recall, precision, label='PR Curve')
    plt.scatter(recall[best_idx], precision[best_idx], color='red',
                label=f'Best (F1={best_f1:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(args.output_plot, dpi=300)
    logging.info(f"Saved PR curve plot to {args.output_plot}")

    config = {'distance_threshold': float(
        distance_threshold), 'distance_variance': dist_variance}
    with args.config.open('w') as f:
        yaml.dump(config, f)
    logging.info(f"Saved threshold config to {args.config}")


if __name__ == '__main__':
    main()
