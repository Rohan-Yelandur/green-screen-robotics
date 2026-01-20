# Green Screen Replacement Script
# Usage: python green_screen.py <input_h5_file_or_folder> <background_image_or_folder> [--camera all|third|gripper]

import cv2
import numpy as np
import h5py
import argparse
import os

# Global HSV Configuration
LOWER_HSV = np.array([35, 60, 100])
UPPER_HSV = np.array([85, 255, 255])
MORPH_KERNEL = np.ones((3,3), np.uint8)
DEPTH_THRESHOLD = 1500
USE_DEPTH_MASKING = False

def get_mask(frame, lower_hsv, upper_hsv, depth_frame=None):
    """Unified mask function: Color Hysteresis + Optional Depth Cutoff."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # --- HYSTERESIS START ---
    loose_lower = np.clip(lower_hsv - 10, 0, 255)
    loose_upper = np.clip(upper_hsv + 10, 0, 255)
    mask_strict = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_loose = cv2.inRange(hsv, loose_lower, loose_upper)
    mask_strict_dilated = cv2.dilate(mask_strict, MORPH_KERNEL, iterations=1)
    mask = cv2.bitwise_and(mask_loose, mask_strict_dilated)
    # --- HYSTERESIS END ---
    
    # Morphological Closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    
    # Denoise
    mask = cv2.medianBlur(mask, 3)
    
    # --- DEPTH LOGIC START ---
    if depth_frame is not None and USE_DEPTH_MASKING:
        if depth_frame.shape[:2] == mask.shape[:2]:
            is_background = (depth_frame > DEPTH_THRESHOLD).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, is_background)
    # --- DEPTH LOGIC END ---
    
    return mask

def process_video(name, in_ds, out_ds, bg_img, lower_hsv, upper_hsv, depth_ds=None):
    print(f"Processing (Masking): {name}")
    
    chunk_size = 100
    num_frames, height, width = in_ds.shape[:3]
    
    # Resize background to match video dimensions
    bg_resized = cv2.resize(bg_img, (width, height))
    
    for i in range(0, num_frames, chunk_size):
        end = min(i + chunk_size, num_frames)
        chunk = in_ds[i:end]
        
        depth_chunk = None
        if depth_ds is not None:
             depth_chunk = depth_ds[i:end]
             
        processed = np.zeros_like(chunk)
        
        for j, frame in enumerate(chunk):
            # Normalize if needed
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Extract depth frame if available
            depth_frame = None
            if depth_chunk is not None:
                depth_frame = depth_chunk[j]

            mask = get_mask(frame, lower_hsv, upper_hsv, depth_frame)
            
            mask_inv = cv2.bitwise_not(mask)
            fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            bg = cv2.bitwise_and(bg_resized, bg_resized, mask=mask)
            processed[j] = cv2.add(fg, bg)
        
        out_ds[i:end] = processed
        print(f"  {end}/{num_frames}", end='\r')
    print("")

def process_file(input_path, background_path, output_path):
    bg_img = cv2.imread(background_path)
    if bg_img is None:
        print(f"Error: Could not load {background_path}")
        return
    
    with h5py.File(input_path, 'r') as infile, h5py.File(output_path, 'w') as outfile:
        for key, val in infile.attrs.items():
            outfile.attrs[key] = val

        def visit_item(name, obj):
            if isinstance(obj, h5py.Group):
                if name not in outfile:
                    outfile.create_group(name)
            elif isinstance(obj, h5py.Dataset):
                path_lower = name.lower()
                is_gripper = any(k in path_lower for k in ['gripper', 'eye_in_hand', 'wrist'])
                is_third = any(k in path_lower for k in ['agentview', 'third', 'external', 'front', 'primary'])
                is_video = len(obj.shape) >= 3
                
                # Exclude depth maps from being processed as color video
                is_depth = 'depth' in path_lower
                
                should_mask = is_third and not is_gripper and not is_depth
                if is_video and should_mask:
                    # Apply Green Screen Logic
                    print(f"Processing (Masking): {name}")
                    # Try to find associated depth data
                    # Assuming naming convention: 'image' -> 'depth'
                    depth_ds = None
                    if 'image' in name:
                        depth_name = name.replace('image', 'depth')
                        if depth_name in infile:
                             depth_ds = infile[depth_name]
                             print(f"  + Using Depth Selection: {depth_name}")

                    # Create dataset in output file
                    ds = outfile.create_dataset(name, shape=obj.shape, dtype=obj.dtype, chunks=True, compression='gzip')
                    for k, v in obj.attrs.items(): ds.attrs[k] = v
                    process_video(name, obj, ds, bg_img, LOWER_HSV, UPPER_HSV, depth_ds)
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

    args = parser.parse_args()
    
    if not args.background_path:
        parser.error("the following arguments are required: background_path")
    

    
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
            
            process_file(src_path, bg_path, dst_path)