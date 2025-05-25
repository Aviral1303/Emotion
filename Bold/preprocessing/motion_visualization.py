#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion Data Visualization Tool

This script provides visualization utilities for motion capture data,
including skeleton animation and motion heatmaps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import re
from typing import List, Dict, Tuple, Optional, Union
from emotion_recognition_pipeline import parse_bvh_file, normalize_motion_data, EMOTION_MAP, parse_filename

try:
    import bvh
except ImportError:
    print("Installing bvh library...")
    import subprocess
    subprocess.check_call(["pip", "install", "bvh"])
    import bvh


def plot_joint_trajectory(motion_data: np.ndarray, joint_index: int, title: str = None) -> None:
    """
    Plot the trajectory of a specific joint over time.
    
    Args:
        motion_data: Array of shape [timesteps, num_joints*3]
        joint_index: Index of the joint to visualize
        title: Optional title for the plot
    """
    # Extract x, y, z coordinates for the joint
    x = motion_data[:, joint_index*3]
    y = motion_data[:, joint_index*3 + 1]
    z = motion_data[:, joint_index*3 + 2]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(x, y, z, 'b-', linewidth=2)
    
    # Mark start and end points
    ax.plot([x[0]], [y[0]], [z[0]], 'go', markersize=10, label='Start')
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'ro', markersize=10, label='End')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Joint {joint_index} Trajectory')
    
    ax.legend()
    plt.tight_layout()
    
    return fig


def create_motion_heatmap(motion_data: np.ndarray, title: str = None) -> None:
    """
    Create a heatmap visualization of motion data.
    
    Args:
        motion_data: Array of shape [timesteps, num_joints*3]
        title: Optional title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize data for better visualization
    normalized_data = (motion_data - np.min(motion_data)) / (np.max(motion_data) - np.min(motion_data) + 1e-8)
    
    # Custom colormap (blue to red)
    cmap = LinearSegmentedColormap.from_list('motion_cmap', ['blue', 'white', 'red'], N=256)
    
    # Create heatmap
    im = ax.imshow(normalized_data, aspect='auto', cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Normalized Motion Value')
    
    # Set labels and title
    ax.set_xlabel('Joint Position Features')
    ax.set_ylabel('Time Frame')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Motion Heatmap')
    
    plt.tight_layout()
    
    return fig


def animate_skeleton(bvh_filepath: str, downsample_factor: int = 4, save_path: Optional[str] = None) -> None:
    """
    Create a skeleton animation from a BVH file.
    
    Args:
        bvh_filepath: Path to the BVH file
        downsample_factor: Factor by which to downsample frames (higher = faster animation)
        save_path: Optional path to save the animation (as .mp4)
    """
    with open(bvh_filepath) as f:
        mocap = bvh.Bvh(f.read())
    
    # Get number of frames and joints
    num_frames = mocap.nframes
    joints = list(mocap.get_joints())
    num_joints = len(joints)
    
    # Initialize the array for positions
    positions = np.zeros((num_frames, num_joints, 3))
    
    # Extract joint positions for each frame
    for frame_idx in range(num_frames):
        for joint_idx, joint in enumerate(joints):
            # Get the joint position for this frame
            try:
                x, y, z = mocap.frame_joint_position(frame_idx, joint)
                positions[frame_idx, joint_idx] = [x, y, z]
            except:
                # Some joints might not have positions
                positions[frame_idx, joint_idx] = [0, 0, 0]
    
    # Get filename metadata
    try:
        metadata = parse_filename(bvh_filepath)
        title = f"Actor: {metadata['actor_id']} | Emotion: {metadata['emotion']} | Take: {metadata['take_id']}"
    except:
        title = os.path.basename(bvh_filepath)
    
    # Create connections based on parent-child relationships in the hierarchy
    connections = []
    for i, joint in enumerate(joints):
        try:
            parent = mocap.joint_parent(joint)
            if parent in joints:  # Skip root joint
                parent_idx = joints.index(parent)
                connections.append((parent_idx, i))
        except LookupError:
            # Skip if parent can't be found
            continue
    
    # If no connections found, create artificial connections based on joint order
    if not connections and num_joints > 1:
        print("No parent-child connections found, creating artificial connections based on joint order")
        for i in range(1, num_joints):
            connections.append((i-1, i))
    
    # Downsample frames for faster animation
    frames_to_show = range(0, num_frames, downsample_factor)
    
    # Create the animation figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize empty line objects for each bone
    lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in connections]
    
    # Set axis limits based on the motion data
    min_val = np.min(positions)
    max_val = np.max(positions)
    
    buffer = (max_val - min_val) * 0.1  # Add 10% buffer
    ax.set_xlim([min_val - buffer, max_val + buffer])
    ax.set_ylim([min_val - buffer, max_val + buffer])
    ax.set_zlim([min_val - buffer, max_val + buffer])
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Function to update the animation at each frame
    def update(frame_idx):
        frame = frames_to_show[frame_idx]
        pos = positions[frame]
        
        # Update each line (bone)
        for i, (start_idx, end_idx) in enumerate(connections):
            start_pos = pos[start_idx]
            end_pos = pos[end_idx]
            lines[i].set_data([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]])
            lines[i].set_3d_properties([start_pos[2], end_pos[2]])
        
        # Update title to show current frame
        ax.set_title(f"{title} | Frame: {frame}/{num_frames}")
        return lines
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_to_show), interval=50, blit=True
    )
    
    if save_path:
        # Save animation as mp4
        try:
            writer = animation.FFMpegWriter(fps=30)
            anim.save(save_path, writer=writer, dpi=100)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            plt.close()
            return
        plt.close()
    else:
        # Display the animation
        plt.tight_layout()
        plt.show()
    
    return anim


def compare_emotions(data_dir: str, actor_id: str, emotions: List[str], 
                     output_dir: str = './visualization') -> None:
    """
    Compare motion patterns across different emotions for the same actor.
    
    Args:
        data_dir: Directory containing BVH files
        actor_id: ID of the actor to use for comparison (e.g., 'F01')
        emotions: List of emotions to compare
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the first take of each emotion for the specified actor
    motion_data_dict = {}
    
    for emotion in emotions:
        # Look for files matching the pattern (e.g., F01A0V1.bvh)
        emotion_code = None
        for key, value in {'A': 'Angry', 'D': 'Disgust', 'F': 'Fearful', 
                           'H': 'Happy', 'N': 'Neutral', 'SA': 'Sad', 
                           'SU': 'Surprise'}.items():
            if value == emotion:
                emotion_code = key
                break
        
        if emotion_code:
            # Look for files starting with the actor and emotion code
            matching_files = [f for f in os.listdir(data_dir) 
                             if f.startswith(f"{actor_id}{emotion_code}")]
            
            if matching_files:
                filepath = os.path.join(data_dir, matching_files[0])
                motion_data = parse_bvh_file(filepath)
                motion_data_dict[emotion] = normalize_motion_data(motion_data)
    
    if not motion_data_dict:
        print(f"No motion data found for actor {actor_id}")
        return
    
    # Plot joint trajectories
    fig, axes = plt.subplots(2, len(motion_data_dict), 
                            figsize=(5*len(motion_data_dict), 10), 
                            subplot_kw={'projection': '3d'})
    
    if len(motion_data_dict) == 1:
        axes = np.array([axes[0], axes[1]])
    
    axes = axes.flatten()
    
    for i, (emotion, data) in enumerate(motion_data_dict.items()):
        # Plot first joint (usually hip/root)
        ax = axes[i]
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        
        ax.plot(x, y, z, 'b-', linewidth=2)
        ax.set_title(f'Root Joint - {emotion}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Plot another informative joint (e.g., right hand - joint index may need adjustment)
        hand_joint_idx = min(5, data.shape[1]//3 - 1)  # Choose a reasonable joint index
        ax = axes[i + len(motion_data_dict)]
        x = data[:, hand_joint_idx*3]
        y = data[:, hand_joint_idx*3 + 1]
        z = data[:, hand_joint_idx*3 + 2]
        
        ax.plot(x, y, z, 'r-', linewidth=2)
        ax.set_title(f'Hand Joint - {emotion}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"actor_{actor_id}_emotion_comparison.png"))
    plt.close()
    
    # Create heatmaps for each emotion
    for emotion, data in motion_data_dict.items():
        fig = create_motion_heatmap(data, title=f"Motion Heatmap - Actor {actor_id}, {emotion}")
        plt.savefig(os.path.join(output_dir, f"actor_{actor_id}_{emotion}_heatmap.png"))
        plt.close(fig)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Motion Data Visualization')
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Directory containing BVH files')
    parser.add_argument('--output_dir', type=str, default='./visualization',
                        help='Directory to save visualizations')
    parser.add_argument('--file', type=str, help='Specific BVH file to visualize')
    parser.add_argument('--actor_id', type=str, default='F01', 
                        help='Actor ID for emotion comparison')
    parser.add_argument('--animate', action='store_true', 
                        help='Create skeleton animation')
    parser.add_argument('--compare', action='store_true',
                        help='Compare emotions for a specific actor')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.file:
        # Visualize a specific file
        filepath = args.file if os.path.exists(args.file) else os.path.join(args.data_dir, args.file)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            exit(1)
            
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(filepath))[0]
        
        # Parse and normalize motion data
        motion_data = parse_bvh_file(filepath)
        motion_data = normalize_motion_data(motion_data)
        
        # Create and save motion heatmap
        fig = create_motion_heatmap(motion_data, title=f"Motion Heatmap - {filename}")
        plt.savefig(os.path.join(args.output_dir, f"{filename}_heatmap.png"))
        plt.close(fig)
        
        # Plot trajectory of root joint
        fig = plot_joint_trajectory(motion_data, 0, title=f"Root Joint Trajectory - {filename}")
        plt.savefig(os.path.join(args.output_dir, f"{filename}_root_trajectory.png"))
        plt.close(fig)
        
        if args.animate:
            # Create skeleton animation
            animate_skeleton(
                filepath,
                save_path=os.path.join(args.output_dir, f"{filename}_animation.mp4")
            )
    
    elif args.compare:
        # Compare emotions for the specified actor
        emotions = list(EMOTION_MAP.keys())
        compare_emotions(args.data_dir, args.actor_id, emotions, args.output_dir)
    
    else:
        print("Please specify a file to visualize with --file or use --compare to compare emotions.") 