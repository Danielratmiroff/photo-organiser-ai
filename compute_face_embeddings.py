#!/usr/bin/env python3
"""
compute_face_embeddings.py

Detect faces in an image, compute 128-dimensional embeddings for each face,
and save the embeddings along with image IDs to a CSV text file.

Requires:
    pip install face_recognition
    pip install git+https://github.com/ageitgey/face_recognition_models
"""

import sys
import argparse
import csv
import logging
from pathlib import Path
import face_recognition


def main():
    parser = argparse.ArgumentParser(
        description="Detect faces and compute 128-dimensional embeddings for images"
    )
    parser.add_argument(
        'input_path', type=Path,
        help="Path to an image file or directory containing images"
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=Path('face_embeddings.csv'),
        help="Path to the output CSV file"
    )
    parser.add_argument(
        '-l', '--identity', type=str, default=None,
        help="Identity label to assign to all detected faces (defaults to parent directory name)"
    )
    parser.add_argument(
        '-r', '--recursive', action='store_true',
        help="Recursively scan subdirectories if input_path is a directory"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    input_path = args.input_path
    if not (input_path.is_file() or input_path.is_dir()):
        logging.error(
            f"Input path '{input_path}' is not a valid file or directory.")
        sys.exit(1)

    # Gather image files
    patterns = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    if input_path.is_file():
        image_files = [input_path]
    else:
        if args.recursive:
            image_files = [
                p for pattern in patterns for p in input_path.rglob(pattern)]
        else:
            image_files = [p for p in input_path.iterdir() if p.is_file(
            ) and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    image_files.sort()
    if not image_files:
        logging.error(f"No image files found under '{input_path}'")
        sys.exit(1)

    logging.info(f"Processing {len(image_files)} image(s) from {input_path}")

    # Prepare CSV output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['image_id', 'face_id', 'face_index',
                  'identity'] + [f'emb_{i}' for i in range(128)]
        writer.writerow(header)
        # Process each image
        for image_path in image_files:
            image_id = image_path.name
            identity = args.identity or image_path.parent.name
            logging.info(f"Loading image {image_path}")
            image = face_recognition.load_image_file(str(image_path))

            logging.info("Detecting face embeddings...")
            embeddings = face_recognition.face_encodings(image)

            if not embeddings:
                logging.warning(f"No faces found in {image_path}")

            for idx, emb in enumerate(embeddings):
                face_id = f"{image_path.stem}_{idx}"
                row = [image_id, face_id, idx, identity] + \
                    [f"{v:.6f}" for v in emb]
                writer.writerow(row)

    logging.info(
        f"Saved embeddings for {len(image_files)} image(s) to {args.output}")


if __name__ == '__main__':
    main()
