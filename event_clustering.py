#!/usr/bin/env python3
"""
event_clustering.py

Group photos by known identities by comparing face embeddings to reference embeddings.
Copies photos into output/<identity>/ directories for any matching faces, and optionally into 'unknown'.

Requires:
    pip install face_recognition face_recognition_models numpy pyyaml
"""

import sys
import argparse
import logging
from pathlib import Path
import shutil
import face_recognition
import numpy as np
import yaml


def load_reference_embeddings(known_dir, patterns):
    """Load average face embeddings for each identity from reference images."""
    known_embs = {}
    for identity_dir in sorted(known_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        identity = identity_dir.name
        embs = []
        for pattern in patterns:
            for img_path in identity_dir.glob(pattern):
                image = face_recognition.load_image_file(str(img_path))
                faces = face_recognition.face_encodings(image)
                if faces:
                    embs.append(faces[0])
                else:
                    logging.warning(
                        f"No face found in reference image {img_path} for '{identity}'")
        if embs:
            known_embs[identity] = np.mean(np.stack(embs), axis=0)
            logging.info(
                f"Loaded {len(embs)} reference embedding(s) for '{identity}'")
        else:
            logging.warning(
                f"Skipping identity '{identity}' with no valid embeddings")
    return known_embs


def cluster_photos(test_dir, known_embs, threshold, output_dir, patterns, recursive, include_unknown):
    """Classify test images by comparing face embeddings to known embeddings."""
    # Gather test image paths
    if recursive:
        test_images = [
            p for pattern in patterns for p in test_dir.rglob(pattern)]
    else:
        test_images = [p for p in test_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    test_images = sorted(test_images)
    if not test_images:
        logging.error(f"No test images found in {test_dir}")
        sys.exit(1)
    logging.info(f"Found {len(test_images)} test image(s) in {test_dir}")

    # Process each test image
    for img_path in test_images:
        image = face_recognition.load_image_file(str(img_path))
        faces = face_recognition.face_encodings(image)
        matched = set()
        for emb in faces:
            # Compute distances to each known identity
            dists = {identity: np.linalg.norm(
                emb - ref) for identity, ref in known_embs.items()}
            if not dists:
                break
            # Find best match
            best_id, best_dist = min(dists.items(), key=lambda x: x[1])
            if best_dist <= threshold:
                matched.add(best_id)
        if matched:
            for identity in matched:
                dest = output_dir / identity
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, dest / img_path.name)
            logging.info(
                f"Copied '{img_path.name}' to identities: {', '.join(sorted(matched))}")
        else:
            if include_unknown:
                dest = output_dir / 'unknown'
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, dest / img_path.name)
                logging.info(f"Copied '{img_path.name}' to 'unknown'")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster photos by known identities"
    )
    parser.add_argument('known_dir', type=Path,
                        help="Directory with subfolders per identity containing reference images")
    parser.add_argument('test_dir', type=Path,
                        help="Directory containing test images to cluster")
    parser.add_argument('--config', type=Path, default=Path('face_match_config.yaml'),
                        help="YAML config file with 'distance_threshold'")
    parser.add_argument('-o', '--output', type=Path, default=Path('clustered_photos'),
                        help="Base output directory for clustered photos")
    parser.add_argument('-r', '--recursive', action='store_true',
                        help="Recursively search test_dir for images")
    parser.add_argument('--include-unknown', action='store_true',
                        help="Copy images with no match into 'unknown' subfolder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    # Validate inputs
    if not args.known_dir.is_dir():
        logging.error(f"Known directory '{args.known_dir}' is invalid")
        sys.exit(1)
    if not args.test_dir.is_dir():
        logging.error(f"Test directory '{args.test_dir}' is invalid")
        sys.exit(1)
    if not args.config.is_file():
        logging.error(f"Config file '{args.config}' not found")
        sys.exit(1)

    # Load threshold
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    base_threshold = cfg.get('distance_threshold')
    if base_threshold is None:
        logging.error("'distance_threshold' missing in config file")
        sys.exit(1)
    variance = cfg.get('distance_variance', 0.0)
    # Adjust threshold by adding variance
    threshold = base_threshold + variance
    logging.info(
        f"Using base threshold {base_threshold:.6f} + variance {variance:.6f} = adjusted threshold {threshold:.6f}")

    patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

    # Load reference embeddings
    known_embs = load_reference_embeddings(args.known_dir, patterns)
    if not known_embs:
        logging.error("No reference embeddings loaded; aborting")
        sys.exit(1)

    # Cluster and copy photos
    cluster_photos(
        args.test_dir,
        known_embs,
        threshold,
        args.output,
        patterns,
        args.recursive,
        args.include_unknown
    )


if __name__ == '__main__':
    main()
