"""
DA3 (Depth-Anything-3) Test Script for LV-Bench2

This script demonstrates DA3's depth estimation capabilities for video analysis.
It extracts depth maps from video frames and tracks object motion in 3D space.

Usage:
    python mechanics/da3_test.py --video data/6.mp4 --output-dir mechanics/outputs/da3_test

Features:
    - Monocular depth estimation using DA3-LARGE-1.1
    - Frame-by-frame depth map extraction
    - 3D trajectory visualization combining 2D tracking with depth
    - Optional object segmentation integration with SAM3

Config:
    - Model: ./weights/da3
    - Device: CUDA
    - Output: Depth videos, depth maps, 3D visualizations
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add Depth-Anything-3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-3', 'src'))

from depth_anything_3.api import DepthAnything3


def extract_frames(video_path, max_frames=100, stride=1):
    """
    Extract frames from video for depth estimation.

    Args:
        video_path: Path to input video
        max_frames: Maximum number of frames to process
        stride: Frame sampling stride (1 = every frame)

    Returns:
        frames: List of RGB frames (numpy arrays)
        fps: Video frame rate
        frame_indices: List of frame indices
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_indices = []
    frame_idx = 0

    print(f"Video info: {total_frames} frames @ {fps:.2f} fps")

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_indices.append(frame_idx)

        frame_idx += 1

    cap.release()

    print(f"Extracted {len(frames)} frames (stride={stride})")
    return frames, fps, frame_indices


def visualize_depth_maps(depth_maps, frame_indices, output_dir, fps):
    """
    Create visualization video of depth maps.

    Args:
        depth_maps: [N, H, W] array of depth values
        frame_indices: List of frame indices
        output_dir: Output directory
        fps: Video frame rate
    """
    output_path = os.path.join(output_dir, "depth_visualization.mp4")

    # Normalize depth for visualization
    depth_norm = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())

    # Apply colormap
    h, w = depth_maps[0].shape
    # Use mp4v first, then re-encode with ffmpeg
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))

    for i, depth in enumerate(depth_norm):
        # Convert to colormap (turbo is good for depth)
        depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

        # Add frame number
        cv2.putText(depth_colored, f"Frame {frame_indices[i]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(depth_colored)

    out.release()

    # Re-encode with ffmpeg for better compatibility
    import subprocess
    try:
        subprocess.run([
            'ffmpeg', '-i', temp_output, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'medium', '-crf', '23', output_path, '-y'
        ], check=True, capture_output=True, text=True)
        os.remove(temp_output)
        print(f"Depth visualization saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: ffmpeg re-encoding failed, using original: {e}")
        os.rename(temp_output, output_path)
        print(f"Depth visualization saved to: {output_path}")


def detect_moving_objects_simple(frames, depth_maps, threshold=30):
    """
    Simple motion detection using frame differencing.
    Returns regions with significant motion for trajectory tracking.

    Args:
        frames: [N, H, W, 3] RGB frames
        depth_maps: [N, H, W] depth maps
        threshold: Motion detection threshold

    Returns:
        motion_masks: [N, H, W] binary masks of moving regions
        centroids: List of (x, y, z) centroids for each frame
    """
    n_frames = len(frames)
    motion_masks = []
    centroids = []

    # Get dimensions
    frame_h, frame_w = frames[0].shape[:2]
    depth_h, depth_w = depth_maps[0].shape

    for i in range(n_frames):
        if i == 0:
            # No motion in first frame
            motion_masks.append(np.zeros(frames[i].shape[:2], dtype=np.uint8))
            centroids.append(None)
            continue

        # Frame difference
        gray_prev = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray_curr, gray_prev)

        # Threshold
        _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        motion_masks.append(motion_mask)

        # Calculate centroid with depth
        if motion_mask.sum() > 0:
            coords = np.argwhere(motion_mask > 0)  # [y, x] format
            center_y, center_x = coords.mean(axis=0)

            # Map coordinates from frame space to depth map space
            depth_y = int(center_y * depth_h / frame_h)
            depth_x = int(center_x * depth_w / frame_w)

            # Clamp to valid range
            depth_y = max(0, min(depth_h - 1, depth_y))
            depth_x = max(0, min(depth_w - 1, depth_x))

            # Get depth at centroid
            depth_value = depth_maps[i][depth_y, depth_x]
            centroids.append((center_x, center_y, depth_value))
        else:
            centroids.append(None)

    return motion_masks, centroids


def visualize_3d_trajectory(centroids, frame_indices, output_dir):
    """
    Create 3D visualization of object trajectory using depth information.

    Args:
        centroids: List of (x, y, depth) tuples
        frame_indices: List of frame indices
        output_dir: Output directory
    """
    # Filter out None values
    valid_points = [(i, c) for i, c in enumerate(centroids) if c is not None]

    if len(valid_points) < 2:
        print("Not enough valid centroids for 3D trajectory visualization")
        return

    indices, points = zip(*valid_points)
    xs, ys, depths = zip(*points)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(xs, ys, depths, 'b-', linewidth=2, label='Trajectory')
    ax.scatter(xs, ys, depths, c=range(len(xs)), cmap='viridis', s=50, marker='o')

    # Start and end points
    ax.scatter([xs[0]], [ys[0]], [depths[0]], c='green', s=200, marker='o', label='Start')
    ax.scatter([xs[-1]], [ys[-1]], [depths[-1]], c='red', s=200, marker='*', label='End')

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Depth')
    ax.set_title('3D Object Trajectory (DA3 Depth)')
    ax.legend()

    # Save plot
    output_path = os.path.join(output_dir, "trajectory_3d.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"3D trajectory plot saved to: {output_path}")

    # Also save as interactive HTML (optional)
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = os.path.join(output_dir, "trajectory_3d.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        print(f"3D trajectory PDF saved to: {pdf_path}")
    except Exception as e:
        print(f"Could not save PDF: {e}")

    plt.close()


def create_side_by_side_video(frames, depth_maps, motion_masks, centroids,
                               output_dir, fps, frame_indices):
    """
    Create side-by-side video: RGB + Depth + Motion

    Args:
        frames: [N, H, W, 3] RGB frames
        depth_maps: [N, H, W] depth maps
        motion_masks: [N, H, W] motion masks
        centroids: List of (x, y, depth) tuples
        output_dir: Output directory
        fps: Video frame rate
        frame_indices: List of frame indices
    """
    output_path = os.path.join(output_dir, "combined_visualization.mp4")

    frame_h, frame_w = frames[0].shape[:2]
    depth_h, depth_w = depth_maps[0].shape

    # Normalize depth
    depth_norm = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())

    # Create video writer (3x width for side-by-side, use frame dimensions)
    # Use mp4v first, then re-encode with ffmpeg
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_w * 3, frame_h))

    for i in range(len(frames)):
        # RGB frame
        rgb_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)

        # Depth frame (colorized) - resize to match frame dimensions
        depth_colored = cv2.applyColorMap((depth_norm[i] * 255).astype(np.uint8),
                                          cv2.COLORMAP_TURBO)
        depth_colored = cv2.resize(depth_colored, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

        # Motion frame (show motion mask as red overlay, not full green)
        motion_frame = rgb_frame.copy()
        # Create red overlay only where motion is detected
        motion_mask_resized = cv2.resize(motion_masks[i], (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        red_overlay = np.zeros_like(motion_frame)
        red_overlay[:, :, 2] = motion_mask_resized  # Red channel in BGR
        motion_frame = cv2.addWeighted(motion_frame, 1.0, red_overlay, 0.3, 0)

        # Draw centroid if available
        if centroids[i] is not None:
            cx, cy, depth_val = centroids[i]
            cv2.circle(motion_frame, (int(cx), int(cy)), 10, (0, 255, 0), -1)
            cv2.circle(motion_frame, (int(cx), int(cy)), 12, (255, 255, 255), 2)
            cv2.putText(motion_frame, f"Depth={depth_val:.2f}",
                       (int(cx) + 20, int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(motion_frame, f"Depth={depth_val:.2f}",
                       (int(cx) + 20, int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Add labels
        cv2.putText(rgb_frame, "RGB", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(depth_colored, "Depth", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(motion_frame, "Motion + Depth", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Concatenate horizontally
        combined = np.hstack([rgb_frame, depth_colored, motion_frame])

        out.write(combined)

    out.release()

    # Re-encode with ffmpeg for better compatibility
    import subprocess
    try:
        subprocess.run([
            'ffmpeg', '-i', temp_output, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'medium', '-crf', '23', output_path, '-y'
        ], check=True, capture_output=True, text=True)
        os.remove(temp_output)
        print(f"Combined visualization saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: ffmpeg re-encoding failed, using original: {e}")
        os.rename(temp_output, output_path)
        print(f"Combined visualization saved to: {output_path}")


def save_results_json(centroids, frame_indices, depth_maps, output_dir):
    """
    Save trajectory and depth data to JSON.

    Args:
        centroids: List of (x, y, depth) tuples
        frame_indices: List of frame indices
        depth_maps: [N, H, W] depth maps
        output_dir: Output directory
    """
    results = {
        "trajectory": [],
        "frame_indices": frame_indices,
        "depth_stats": {
            "min": float(depth_maps.min()),
            "max": float(depth_maps.max()),
            "mean": float(depth_maps.mean()),
            "std": float(depth_maps.std())
        }
    }

    for i, centroid in enumerate(centroids):
        if centroid is not None:
            results["trajectory"].append({
                "frame_index": frame_indices[i],
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "depth": float(centroid[2])
            })
        else:
            results["trajectory"].append({
                "frame_index": frame_indices[i],
                "x": None,
                "y": None,
                "depth": None
            })

    output_path = os.path.join(output_dir, "trajectory_depth.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DA3 Test Script for Video Depth Estimation")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output-dir", type=str, default="mechanics/outputs/da3_test",
                       help="Output directory")
    parser.add_argument("--model-path", type=str,
                       default="./weights/da3",
                       help="Path to DA3 model")
    parser.add_argument("--max-frames", type=int, default=100,
                       help="Maximum number of frames to process")
    parser.add_argument("--stride", type=int, default=1,
                       help="Frame sampling stride")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--motion-threshold", type=int, default=50,
                       help="Motion detection threshold (higher = less sensitive, default=50)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DA3 (Depth-Anything-3) Test Script")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Step 1: Extract frames
    print("\n[1/6] Extracting frames from video...")
    frames, fps, frame_indices = extract_frames(args.video, args.max_frames, args.stride)

    # Step 2: Load DA3 model
    print("\n[2/6] Loading DA3 model...")
    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model_path)
    model = model.to(device=device)
    print(f"Model loaded successfully on {args.device}")

    # Step 3: Run depth estimation
    print("\n[3/6] Running depth estimation...")
    print(f"Processing {len(frames)} frames...")

    # DA3 expects file paths or PIL images, so we save frames temporarily
    temp_dir = os.path.join(args.output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)

    # Run inference
    prediction = model.inference(frame_paths)

    # Extract depth maps
    depth_maps = prediction.depth  # [N, H, W]
    print(f"Depth maps shape: {depth_maps.shape}")
    print(f"Depth range: [{depth_maps.min():.2f}, {depth_maps.max():.2f}]")

    # Clean up temp frames
    import shutil
    shutil.rmtree(temp_dir)

    # Step 4: Detect moving objects
    print("\n[4/6] Detecting moving objects...")
    print(f"Motion detection threshold: {args.motion_threshold}")
    motion_masks, centroids = detect_moving_objects_simple(frames, depth_maps,
                                                           args.motion_threshold)
    valid_centroids = [c for c in centroids if c is not None]
    print(f"Detected motion in {len(valid_centroids)}/{len(centroids)} frames")

    # Calculate motion statistics
    motion_percentages = [(mask.sum() / 255.0 / mask.size * 100) for mask in motion_masks]
    avg_motion = np.mean(motion_percentages)
    print(f"Average motion coverage: {avg_motion:.1f}% of frame")
    if avg_motion > 50:
        print(f"⚠️  WARNING: High motion coverage detected!")
        print(f"   This might be due to camera movement or low threshold.")
        print(f"   Consider increasing --motion-threshold (current: {args.motion_threshold})")

    # Step 5: Create visualizations
    print("\n[5/6] Creating visualizations...")

    # Depth visualization video
    visualize_depth_maps(depth_maps, frame_indices, args.output_dir, fps)

    # 3D trajectory plot
    visualize_3d_trajectory(centroids, frame_indices, args.output_dir)

    # Combined side-by-side video
    create_side_by_side_video(frames, depth_maps, motion_masks, centroids,
                              args.output_dir, fps, frame_indices)

    # Step 6: Save results
    print("\n[6/6] Saving results...")
    save_results_json(centroids, frame_indices, depth_maps, args.output_dir)

    print("\n" + "=" * 60)
    print("DA3 Test Complete!")
    print(f"All results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
