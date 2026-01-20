# H5 Video Viewer
# Usage: python h5_viewer.py <input_h5_file> --camera third|gripper|all
# 'q' to quit, 'p' to pause, 'v' to print pixel values, 'm' to toggle mask

import h5py
import cv2
import numpy as np
import argparse

# Global state for mouse hover
hover_dict = {}
last_hovered_window = None

# Import get_mask and global constants from green_screen to ensure shared logic and values
from green_screen import get_mask, LOWER_HSV, UPPER_HSV, DEPTH_THRESHOLD, USE_DEPTH_MASKING

def mouse_callback(event, x, y, flags, param):
    global last_hovered_window
    if event == cv2.EVENT_MOUSEMOVE:
        hover_dict[param] = (x, y)
        last_hovered_window = param

def view_h5_video(h5_file, camera_view='all'):
    # Initial HSV Configuration from global source
    init_lower = LOWER_HSV
    init_upper = UPPER_HSV
    
    # Callback for trackbars (pass)
    def nothing(x):
        pass

    # Create 'Controls' window early, set default size and resizeable
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 400)
    
    try:
        with h5py.File(h5_file, 'r') as f:
            video_datasets = []
            
            # Explore structure to find videos
            def visit_item(name, obj):
                if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 3:
                    # Filter logic
                    path_lower = name.lower()
                    is_gripper = any(k in path_lower for k in ['gripper', 'eye_in_hand', 'wrist'])
                    is_third = any(k in path_lower for k in ['agentview', 'third', 'external', 'front'])
                    
                    should_include = False
                    if camera_view == 'all': should_include = True
                    elif camera_view == 'gripper' and is_gripper: should_include = True
                    elif camera_view == 'third' and is_third and not is_gripper: should_include = True
                    
                    if should_include:
                         video_datasets.append((name, obj))
            
            f.visititems(visit_item)
            
            if not video_datasets:
                print("No suitable video datasets found.")
                return

            print(f"Found {len(video_datasets)} videos.")
            
            # Load frames
            frames_data = []
            max_frames = 0
            
            # Create a lookup for depth datasets: RGB Name -> Depth Dataset
            depth_lookup = {}
            for name, ds in video_datasets:
                data = ds[:]
                frames_data.append((name, data))
                max_frames = max(max_frames, data.shape[0])
                
                # Check if this is a depth dataset to index it
                # Convention: 'depth' in name.
                # Logic: We actually want to map RGB -> Depth.
                # So we iterate AGAIN or just index everything first?
                # Let's index everything first.
            
            # Re-scan to build link
            # Assumes names like "obs/primary_image" and "obs/primary_depth"
            all_dataset_names = {name: ds for name, ds in video_datasets}
            
            rgb_to_depth_data = {}
            
            for name, _ in frames_data:
                # If this is an RGB image, find its pair
                if 'image' in name: # naive check
                     potential_depth = name.replace('image', 'depth')
                     if potential_depth in all_dataset_names:
                         # Store the actual loaded numpy array, not the H5 dataset (closed later?)
                         # Wait, we loaded all data into 'frames_data'.
                         # Let's find the data in frames_data
                         for n, d in frames_data:
                             if n == potential_depth:
                                 rgb_to_depth_data[name] = d
                                 break
                
            print(f"Playing {max_frames} frames.")
            print("Controls: 'q'=quit, 'p'=pause (or space), 'v'=pixel info, 'm'=toggle mask")
            
            # Initialize trackbars for 'Controls' window now that max_frames is known
            # Playback Controls
            cv2.createTrackbar('Time', 'Controls', 0, max_frames-1, nothing)
            cv2.createTrackbar('Pause', 'Controls', 0, 1, nothing) # 0=Play, 1=Pause
            cv2.createTrackbar('Mask', 'Controls', 0, 1, nothing) # 0=Off, 1=On
            
            # HSV Controls
            cv2.createTrackbar('Low H', 'Controls', init_lower[0], 179, nothing)
            cv2.createTrackbar('Low S', 'Controls', init_lower[1], 255, nothing)
            cv2.createTrackbar('Low V', 'Controls', init_lower[2], 255, nothing)
            cv2.createTrackbar('High H', 'Controls', init_upper[0], 179, nothing)
            cv2.createTrackbar('High S', 'Controls', init_upper[1], 255, nothing)
            cv2.createTrackbar('High V', 'Controls', init_upper[2], 255, nothing)

            # Initialize windows and callbacks
            for name, _ in frames_data:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(name, mouse_callback, name)
            
            idx = 0
            paused = False
            
            # Update 'Time' slider max range if needed (already set above using 0 as placeholder max logic issue?)
            # Re-create trackbar with correct max_frames

            
            while True:
                # Handle Window Closure
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) <= 0 for name, _ in frames_data) or \
                   cv2.getWindowProperty('Controls', cv2.WND_PROP_VISIBLE) <= 0:
                    break
                    
                # 1. Sync Logic: Read Control State
                pause_val = cv2.getTrackbarPos('Pause', 'Controls')
                paused = (pause_val == 1)
                
                mask_val = cv2.getTrackbarPos('Mask', 'Controls')
                show_mask = (mask_val == 1)
                
                slider_idx = cv2.getTrackbarPos('Time', 'Controls')
                
                # If paused, let slider control idx (Scrubbing)
                if paused:
                    if slider_idx != idx:
                        idx = slider_idx
                else:
                    # If running, update slider to match idx (Playback)
                    # Use setTrackbarPos to visualize progress
                    if slider_idx != idx: # Avoid feedback loop fighting
                         cv2.setTrackbarPos('Time', 'Controls', idx)
                    
                # Loop / End Handling
                if idx >= max_frames:
                    idx = 0 # Loop back
                    cv2.setTrackbarPos('Time', 'Controls', 0)
                
                # 2. Read HSV Trackbars
                lh = cv2.getTrackbarPos('Low H', 'Controls')
                ls = cv2.getTrackbarPos('Low S', 'Controls')
                lv = cv2.getTrackbarPos('Low V', 'Controls')
                uh = cv2.getTrackbarPos('High H', 'Controls')
                us = cv2.getTrackbarPos('High S', 'Controls')
                uv = cv2.getTrackbarPos('High V', 'Controls')
                
                lower_hsv = np.array([lh, ls, lv])
                upper_hsv = np.array([uh, us, uv])

                for name, data in frames_data:
                    # Safe indexing
                    safe_idx = min(idx, len(data)-1)
                    frame = data[safe_idx]
                        
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    if len(frame.shape)==2 or frame.shape[2]==1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2]==3:
                        pass
                    
                    frame_display = frame.copy()
                    
                    # --- Mask Overlay ---
                    if show_mask:
                        # Find corresponding depth frame if exists
                        current_depth_frame = None
                        if name in rgb_to_depth_data:
                            d_data = rgb_to_depth_data[name]
                            if idx < len(d_data):
                                current_depth_frame = d_data[idx]
                        
                        mask = get_mask(frame_display, lower_hsv, upper_hsv, current_depth_frame)
                        frame_display[mask > 0] = [0, 0, 255]

                    cv2.imshow(name, frame_display)
                    cv2.setWindowTitle(name, f"{name} ({safe_idx}/{max_frames})")
                
                # Key Handling
                key = cv2.waitKey(33) # Always wait 33ms to allow UI updates
                if key == ord('q'): break
                if key == ord('p') or key == 32: # 'p' or space
                    paused = not paused
                    # Update UI to reflect key toggle
                    cv2.setTrackbarPos('Pause', 'Controls', 1 if paused else 0)
                    
                if key == ord('m'): 
                    # Toggle mask via trackbar
                    new_mask_val = 1 if not show_mask else 0
                    cv2.setTrackbarPos('Mask', 'Controls', new_mask_val)
                    print(f"Mask Overlay: {'ON' if new_mask_val else 'OFF'}")
                
                # Check for 'v' key to print pixel info
                if key == ord('v'):
                    for name, data in frames_data:
                        # Only handle the window we are currently hovering over
                        if name == last_hovered_window and name in hover_dict:
                            mx, my = hover_dict[name]
                            
                            safe_idx = min(idx, len(data)-1)
                            raw_frame = data[safe_idx]
                            
                            native_val = "N/A"
                            if 0 <= my < raw_frame.shape[0] and 0 <= mx < raw_frame.shape[1]:
                                native_val = raw_frame[my, mx]

                            if raw_frame.dtype != np.uint8:
                                if raw_frame.max() <= 1.0:
                                    raw_frame = (raw_frame * 255).astype(np.uint8)
                                else:
                                    raw_frame = cv2.normalize(raw_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            if len(raw_frame.shape)==2 or raw_frame.shape[2]==1:
                                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
                                
                            if 0 <= my < raw_frame.shape[0] and 0 <= mx < raw_frame.shape[1]:
                                bgr_val = raw_frame[my, mx]
                                pixel_1x1 = np.uint8([[bgr_val]])
                                hsv_val = cv2.cvtColor(pixel_1x1, cv2.COLOR_BGR2HSV)[0][0]
                                
                                # If native is scalar (depth), print simply. If array (color), print array
                                if np.isscalar(native_val) or (isinstance(native_val, np.ndarray) and native_val.size == 1):
                                     print(f"[{name}] Depth/Val: {native_val} | Display BGR: {bgr_val}")
                                else:
                                     print(f"[{name}] RGB/Native: {native_val} | HSV: {hsv_val}")
                
                if not paused: 
                    idx += 1
                
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_h5')
    parser.add_argument('--camera', default='all', choices=['all', 'third', 'gripper'])
    args = parser.parse_args()
    
    view_h5_video(args.input_h5, args.camera)