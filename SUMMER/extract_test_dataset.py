#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Dataset Extraction Utility

This script extracts a subset of BVH files from the Kinematic Dataset of Actors Expressing Emotions
to a test data directory for validating the emotion recognition pipeline.
"""

import os
import shutil
import glob
from tqdm import tqdm
import random
import argparse

# Source and destination directories
SOURCE_DIR = './data/BVH'
DEST_DIR = './test_data'

def extract_test_bvh_files(samples_per_emotion):
    """
    Extract a subset of BVH files for testing.
    
    Args:
        samples_per_emotion: Number of samples to extract per emotion class
    """
    # Create destination directory if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Clean existing files
    for file in os.listdir(DEST_DIR):
        file_path = os.path.join(DEST_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Get list of all actor directories
    actor_dirs = [d for d in os.listdir(SOURCE_DIR) 
                  if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"Found {len(actor_dirs)} actor directories")
    
    # Dictionary to track files by emotion
    emotion_files = {
        'A': [],    # Angry
        'D': [],    # Disgust
        'F': [],    # Fearful
        'H': [],    # Happy
        'N': [],    # Neutral
        'SA': [],   # Sad
        'SU': []    # Surprise
    }
    
    # Find files for each emotion
    for actor in actor_dirs:
        actor_dir = os.path.join(SOURCE_DIR, actor)
        bvh_files = glob.glob(os.path.join(actor_dir, '*.bvh'))
        
        for bvh_file in bvh_files:
            filename = os.path.basename(bvh_file)
            # Extract emotion code from filename (e.g., F01A0V1.bvh -> A)
            if filename[3:5] in ['SA', 'SU']:
                emotion_code = filename[3:5]
            else:
                emotion_code = filename[3]
                
            if emotion_code in emotion_files:
                emotion_files[emotion_code].append(bvh_file)
    
    # Sample files from each emotion
    selected_files = []
    for emotion, files in emotion_files.items():
        if files:
            # Take a random sample of files for this emotion
            sample_size = min(samples_per_emotion, len(files))
            selected = random.sample(files, sample_size)
            selected_files.extend(selected)
            print(f"Selected {sample_size} files for emotion {emotion}")
    
    print(f"Total files selected: {len(selected_files)}")
    
    # Copy selected files to test directory
    for bvh_file in tqdm(selected_files, desc="Copying BVH files"):
        dest_file = os.path.join(DEST_DIR, os.path.basename(bvh_file))
        shutil.copy2(bvh_file, dest_file)
    
    print(f"Successfully copied {len(selected_files)} BVH files to {DEST_DIR}")
    
    # Analyze emotion distribution
    emotion_counts = {}
    for filename in os.listdir(DEST_DIR):
        if filename.endswith('.bvh'):
            # Extract emotion code from filename
            if filename[3:5] in ['SA', 'SU']:
                emotion_code = filename[3:5]
            else:
                emotion_code = filename[3]
                
            # Map code to emotion name
            emotion_map = {
                'A': 'Angry',
                'D': 'Disgust',
                'F': 'Fearful',
                'H': 'Happy',
                'N': 'Neutral',
                'SA': 'Sad',
                'SU': 'Surprise'
            }
            
            emotion = emotion_map.get(emotion_code, 'Unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print("\nTest dataset emotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count} files")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract test dataset with specified samples per emotion')
    parser.add_argument('--samples', type=int, default=3,
                      help='Number of samples per emotion class')
    args = parser.parse_args()
    
    extract_test_bvh_files(args.samples) 