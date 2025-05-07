#!/usr/bin/env python3
"""Extract EXIF timestamp and GPS coordinates from images in a directory and save them to a CSV file."""

import sys
import argparse
import csv
import logging
from pathlib import Path
from functools import lru_cache
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
import exifread

geolocator = Nominatim(user_agent="extract_photo_exif/1.0")


@lru_cache(maxsize=128)
def reverse_geocode(lat, lon):
    """Reverse geocode latitude and longitude to get the city name."""
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        address = location.raw.get('address', {})
        city = address.get('city') or address.get('town') or address.get(
            'village') or address.get('municipality')
        if not city:
            city = address.get('county') or address.get(
                'state') or address.get('region') or ''
        return city
    except GeocoderServiceError as e:
        logging.warning(f"Geocoding failed for {lat}, {lon}: {e}")
        return ''


def convert_to_degrees(value):
    """Convert EXIF GPS coordinates to float degrees."""
    d, m, s = value.values

    def to_float(ratio):
        return float(ratio.num) / float(ratio.den) if hasattr(ratio, 'num') and hasattr(ratio, 'den') else float(ratio)
    return to_float(d) + to_float(m) / 60.0 + to_float(s) / 3600.0


def extract_exif_data(image_path):
    """Extract timestamp and GPS data from a single image. Returns (timestamp_str, latitude, longitude)."""
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)

    timestamp_tag = tags.get(
        'EXIF DateTimeOriginal') or tags.get('Image DateTime')
    timestamp = str(timestamp_tag) if timestamp_tag else ''

    lat = lon = ''
    gps_lat = tags.get('GPS GPSLatitude')
    gps_lat_ref = tags.get('GPS GPSLatitudeRef')
    gps_lon = tags.get('GPS GPSLongitude')
    gps_lon_ref = tags.get('GPS GPSLongitudeRef')
    print(
        f"gps_lat: {gps_lat}, gps_lat_ref: {gps_lat_ref}, gps_lon: {gps_lon}, gps_lon_ref: {gps_lon_ref}")

    if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
        lat_deg = convert_to_degrees(gps_lat)
        if str(gps_lat_ref) != 'N':
            lat_deg = -lat_deg
        lon_deg = convert_to_degrees(gps_lon)
        if str(gps_lon_ref) != 'E':
            lon_deg = -lon_deg
        lat = lat_deg
        lon = lon_deg

    return timestamp, lat, lon


def main():
    parser = argparse.ArgumentParser(
        description="Extract EXIF timestamp and GPS coordinates from images and save to a CSV file."
    )
    parser.add_argument('input_dir', type=Path,
                        help="Directory containing image files")

    parser.add_argument('-o', '--output', type=Path, default=Path('photo_exif_data.csv'),
                        help="Path to output CSV file")

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        logging.error(
            f"Input directory '{args.input_dir}' does not exist or is not a directory.")
        sys.exit(1)

    patterns = ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG')

    files = [
        p for pattern in patterns for p in args.input_dir.rglob(pattern)]

    files.sort()
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    logging.info(f"Processing {len(files)} image(s) from {args.input_dir}")

    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['filename', 'timestamp', 'latitude', 'longitude', 'city'])
        for image_path in files:
            timestamp, lat, lon = extract_exif_data(image_path)
            if lat != '' and lon != '':
                city = reverse_geocode(lat, lon)
                time.sleep(1)
            else:
                city = ''
            writer.writerow([str(image_path), timestamp, lat, lon, city])

    logging.info(f"Saved data to {args.output}")


if __name__ == '__main__':
    main()
