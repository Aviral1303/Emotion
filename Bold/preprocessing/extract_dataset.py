#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Extraction Utility

This script extracts all BVH files from the Kinematic Dataset of Actors Expressing Emotions
to a flat data directory for processing by the emotion recognition pipeline.
"""

import os
import shutil
import glob
from tqdm import tqdm

# Source and destination directories
SOURCE_DIR = './kinematic-dataset-of-actors-expressing-emotions-2.1.0/BVH'
DEST_DIR = './data'

def extract_bvh_files():
    """
    Extract all BVH files from the dataset folder structure to the data folder.
    """
    # Create destination directory if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Get list of all actor directories
    actor_dirs = [d for d in os.listdir(SOURCE_DIR) 
                  if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"Found {len(actor_dirs)} actor directories")
    
    # Count total BVH files
    total_files = sum(len(glob.glob(os.path.join(SOURCE_DIR, actor, '*.bvh'))) 
                      for actor in actor_dirs)
    
    print(f"Found {total_files} BVH files to copy")
    
    # Copy all BVH files to the data directory
    copied_files = 0
    
    with tqdm(total=total_files, desc="Copying BVH files") as pbar:
        for actor in actor_dirs:
            actor_dir = os.path.join(SOURCE_DIR, actor)
            bvh_files = glob.glob(os.path.join(actor_dir, '*.bvh'))
            
            for bvh_file in bvh_files:
                dest_file = os.path.join(DEST_DIR, os.path.basename(bvh_file))
                shutil.copy2(bvh_file, dest_file)
                copied_files += 1
                pbar.update(1)
    
    print(f"Successfully copied {copied_files} BVH files to {DEST_DIR}")
    
    # Analyze emotion distribution
    emotion_counts = {}
    for filename in os.listdir(DEST_DIR):
        if filename.endswith('.bvh'):
            # Extract emotion code from filename (e.g., F01A0V1.bvh -> A)
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
    
    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count} files")

if __name__ == "__main__":
    extract_bvh_files() 