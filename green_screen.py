# Green Screen Replacement Script
# Usage: python green_screen.py <input_h5_file_or_folder> <background_image_or_folder> [--camera all|third|gripper]

import cv2
import numpy as np
import h5py
import argparse
import os

# Global HSV Configuration
LOWER_HSV = np.array([35, 60, 50])
UPPER_HSV = np.array([85, 255, 255])

def get_mask(frame, lower_hsv, upper_hsv):
    """Creates a binary mask for green pixels using HSV with refinement."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Morphological Closing (Fill small holes in the foreground/background)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Denoise
    mask = cv2.medianBlur(mask, 5)
    return mask

def process_video(name, in_ds, out_ds, bg_img, lower_hsv, upper_hsv):
    """Processes a video dataset: applies green screen mask with feathering."""
    print(f"Processing (Masking): {name}")
    
    chunk_size = 100
    num_frames, height, width = in_ds.shape[:3]
    
    # Resize background to match video dimensions
    bg_resized = cv2.resize(bg_img, (width, height))
    bg_float = bg_resized.astype(float)
    
    for i in range(0, num_frames, chunk_size):
        end = min(i + chunk_size, num_frames)
        chunk = in_ds[i:end]
        processed = np.zeros_like(chunk)
        
        for j, frame in enumerate(chunk):
            # Normalize if needed
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            mask = get_mask(frame, lower_hsv, upper_hsv)
            
            # Simple Masking
            mask_inv = cv2.bitwise_not(mask)
            fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            bg = cv2.bitwise_and(bg_resized, bg_resized, mask=mask)
            processed[j] = cv2.add(fg, bg)

            # OR

            # Feathering (Soft Edges)
            # mask_soft = cv2.GaussianBlur(mask, (13, 13), 0)
            
            # # Alpha Blending
            # alpha = mask_soft.astype(float) / 255.0
            # alpha = cv2.merge([alpha, alpha, alpha])
            
            # frame_float = frame.astype(float)
            # combined = frame_float * (1.0 - alpha) + bg_float * alpha
            # processed[j] = combined.astype(np.uint8)
        
        out_ds[i:end] = processed
        print(f"  {end}/{num_frames}", end='\r')
    print("")

def process_file(input_path, background_path, output_path, camera_filter='all'):
    bg_img = cv2.imread(background_path)
    if bg_img is None:
        print(f"Error: Could not load {background_path}")
        return
    
    with h5py.File(input_path, 'r') as infile, h5py.File(output_path, 'w') as outfile:
        # Copy file-level attributes
        for key, val in infile.attrs.items():
            outfile.attrs[key] = val

        def visit_item(name, obj):
            if isinstance(obj, h5py.Group):
                if name not in outfile:
                    outfile.create_group(name)
            elif isinstance(obj, h5py.Dataset):
                # Check for camera filter
                path_lower = name.lower()
                is_gripper = any(k in path_lower for k in ['gripper', 'eye_in_hand', 'wrist'])
                is_third = any(k in path_lower for k in ['agentview', 'third', 'external', 'front'])
                
                should_process = False
                if camera_filter == 'all': should_process = True
                elif camera_filter == 'gripper' and is_gripper: should_process = True
                elif camera_filter == 'third' and is_third and not is_gripper: should_process = True
                
                # Check if it's a video (3D array)
                is_video = len(obj.shape) >= 3
                
                # MASKING LOGIC: Only mask third-person cameras
                should_mask = is_third and not is_gripper
                
                if is_video and should_mask:
                    # Apply Green Screen Logic
                    print(f"Processing (Masking): {name}")
                    # Create dataset in output file
                    ds = outfile.create_dataset(name, shape=obj.shape, dtype=obj.dtype, chunks=True, compression='gzip')
                    for k, v in obj.attrs.items(): ds.attrs[k] = v
                    process_video(name, obj, ds, bg_img, LOWER_HSV, UPPER_HSV)
                else:
                    # Copy as is
                    print(f"Copying (Raw): {name}")
                    infile.copy(name, outfile)

        try:
            infile.visititems(visit_item)
        except SystemExit:
            pass 
        print("\nDone.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Input H5 file or directory")
    parser.add_argument('background_path', nargs='?', help="Background image or directory") 
    parser.add_argument('--camera', default='third', choices=['all', 'third', 'gripper'])
    args = parser.parse_args()
    
    if not args.background_path:
        parser.error("the following arguments are required: background_path")
    
    import os
    
    # --- Input Discovery ---
    # 1. Video Files
    if os.path.isdir(args.input_path):
        input_dir = args.input_path
        h5_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.h5')])
        # Base output for batch input is input_dir + _outputs (plural)
        output_root = args.input_path.rstrip('/\\') + '_outputs'
        if not h5_files:
            print(f"No H5 files found in {input_dir}")
            exit()
    else:
        input_dir = os.path.dirname(args.input_path) or '.'
        h5_files = [os.path.basename(args.input_path)]
        # Base output for single file
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_root = base_name + '_outputs'

    # 2. Background Files
    if os.path.isdir(args.background_path):
        bg_dir = args.background_path
        bg_files = sorted([f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if not bg_files:
            print(f"No image files found in {bg_dir}")
            exit()
    else:
        bg_dir = os.path.dirname(args.background_path) or '.'
        bg_files = [os.path.basename(args.background_path)]

    # --- Processing Loop ---
    print(f"Output Root Directory: {output_root}")
    
    total_videos = len(h5_files)
    total_bgs = len(bg_files)
    
    for i, h5_file in enumerate(h5_files):
        video_basename = os.path.splitext(h5_file)[0]
        
        # Create subfolder for this video: <name>_output
        video_out_dir = os.path.join(output_root, video_basename + '_output')
        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)
            
        src_path = os.path.join(input_dir, h5_file)
        
        for j, bg_file in enumerate(bg_files):
            bg_basename = os.path.splitext(bg_file)[0]
            
            # e.g. demo_0_kitchen_output.h5
            final_name = f"{video_basename}_{bg_basename}_output.h5"
            dst_path = os.path.join(video_out_dir, final_name)
            bg_path = os.path.join(bg_dir, bg_file)
            
            # Simpler progress print
            print(f"[{i+1}/{total_videos}] {h5_file} + {bg_file} -> {os.path.basename(video_out_dir)}/{final_name}")
            
            process_file(src_path, bg_path, dst_path, args.camera)