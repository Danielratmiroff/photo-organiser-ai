#!/usr/bin/env python3
"""
A CLI tool that reads an image file, encodes it in base64, and sends it to a local
Ollama Gemma3 model via HTTP to get a textual description.
"""
import argparse
import base64
import sys
import requests
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description='Describe an image using a local Ollama Gemma3 model.')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument(
        '--prompt',
        default='Describe this image.',
        help='Prompt to send to the model')
    parser.add_argument(
        '--model',
        default='gemma3:12b',
        help='Name of the Ollama model to use')
    parser.add_argument(
        '--endpoint',
        default=None,
        help='Ollama API endpoint URL (overrides OLLAMA_HOST env var)')
    args = parser.parse_args()

    # Determine which endpoint to use
    if args.endpoint:
        endpoint = args.endpoint
    else:
        host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        endpoint = f'http://{host}/api/generate'

    # Read and encode the image
    try:
        with open(args.image_path, 'rb') as f:
            img_bytes = f.read()
    except Exception as e:
        print(f'Error reading image file: {e}', file=sys.stderr)
        sys.exit(1)

    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    # Build request payload
    payload = {
        'model': args.model,
        'prompt': args.prompt,
        'images': [img_b64],
        'stream': False
    }

    # Send request
    try:
        resp = requests.post(endpoint, json=payload)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(
            f'Error sending request to Ollama API at {endpoint}: {e}', file=sys.stderr)
        sys.exit(1)

    # Parse response
    try:
        data = resp.json()
    except ValueError:
        print(f'Invalid JSON response: {resp.text}', file=sys.stderr)
        sys.exit(1)

    if isinstance(data, dict):
        if 'response' in data:
            print(data['response'].strip())
            return
    # Fallback to raw JSON dump
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
